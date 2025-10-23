import bisect
import io
import time
import lz4.frame
import numpy as np
import torch
from PIL import Image
import PIL
import concurrent.futures


def compress_image(img, level=9) -> bytes:
    if type(img) == PIL.PngImagePlugin.PngImageFile:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img.format or "PNG")
        img_byte_data = img_byte_arr.getvalue()
    elif type(img) == np.ndarray:
        img_byte_data = img.tobytes()
    else:
        raise ValueError("Unsupported image type.")

    compressed_data = lz4.frame.compress(img_byte_data, compression_level=level, store_size=True)
    return compressed_data


def decompress_image(compressed_data: bytes, output_type="Image", dtype="uint8", shape=None) -> Image:
    decompressed_data = lz4.frame.decompress(compressed_data)
    if output_type == "Image":
        img = Image.open(io.BytesIO(decompressed_data))
    elif output_type == "np":
        img = np.frombuffer(decompressed_data, dtype=np.uint8 if dtype == "uint8" else np.float32)
        if shape is not None:
            img = img.reshape(shape)
    else:
        raise ValueError("Unsupported output type.")
    return img


def compress_image_seq(images, level=9, verbose=False):
    """
    Input: [batch, C, H, W]
    """
    concatenated = np.concatenate(images, axis=0).tobytes()
    compressed = lz4.frame.compress(concatenated, compression_level=level)
    if verbose:
        print(f"original size: {len(concatenated)}")
        print(f"compressed size: {len(compressed)}")
    return compressed


def decompress_image_seq(compressed_data, image_shape, batch_size, dtype=np.uint8, verbose=False):
    if verbose:
        st = time.time()
    decompressed = lz4.frame.decompress(compressed_data)
    flat_array = np.frombuffer(decompressed, dtype=dtype)
    if verbose:
        print(f"decompress time: {time.time()-st}")
    return flat_array.reshape(batch_size, *image_shape)


class PrefixSum:
    def __init__(self, max_len):
        self.ar = []
        self.prefix_sum = np.zeros(1, dtype=np.int32)
        self.max_len = max_len
        self.curr_max = 0

    def add(self, val):
        if len(self.prefix_sum) > self.max_len:
            first = self.prefix_sum[1]
            self.prefix_sum -= first
            self.prefix_sum = [0] + self.prefix_sum[1:]
        if len(self.ar) > self.max_len:
            self.ar = self.ar[1:]
        self.ar.append(val)
        self.prefix_sum = np.append(self.prefix_sum, self.prefix_sum[-1] + val)
        self.curr_max = self.prefix_sum[-1]

    def get_range_idx(self, idx):
        """get the range index of the idx-th element"""
        if idx > self.curr_max - 1:
            raise ValueError("Index out of range.")
        return bisect.bisect_right(self.prefix_sum, idx) - 1

    def get_range_relative_idx(self, idx, range_idx):
        """relative index in a range"""
        return idx - self.prefix_sum[range_idx]


def decompress_single_gridmap(args):
    compressed_data, image_shape, episode_len, dtype, episode_relative_idx = args
    gridmap_episode = decompress_image_seq(compressed_data, image_shape, episode_len, dtype=dtype)
    return torch.tensor(gridmap_episode[episode_relative_idx])


def sample_batch(replay_buffer, episode_lens_prefix_sum, train_param: dict, device, compress_epi=False):
    buffer_size = len(replay_buffer["done"])
    indices = range(min(buffer_size, train_param["replay_buffer_size"]))

    sample_indices = np.random.choice(indices, train_param["batch_size"], replace=False)

    if train_param["use_env_encoding"]:
        if train_param["replay_buffer_compress_img"]:
            image_shape = (224, 224) if "maevit" in train_param["env_encoding_model"] else (3, 224, 224)
            if compress_epi:  # compressed by episode
                decompress_args = []
                for idx in sample_indices:
                    episode_idx = episode_lens_prefix_sum.get_range_idx(idx)
                    episode_len = episode_lens_prefix_sum.ar[episode_idx]
                    episode_relative_idx = episode_lens_prefix_sum.get_range_relative_idx(idx, episode_idx)

                    decompress_args.append(
                        (
                            replay_buffer["gridmap_inputs"][episode_idx],
                            image_shape,
                            episode_len,
                            np.float32,
                            episode_relative_idx,
                        )
                    )
                    decompress_args.append(
                        (
                            replay_buffer["next_gridmap_inputs"][episode_idx],
                            image_shape,
                            episode_len,
                            np.float32,
                            episode_relative_idx,
                        )
                    )
                st_decomp = time.time()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = list(executor.map(decompress_single_gridmap, decompress_args))
                # Split results into current and next gridmaps
                grid_maps = results[0::2]  # Even indices
                next_grid_maps = results[1::2]  # Odd indices

            else:  # compressed by frames
                grid_maps = [
                    torch.tensor(
                        decompress_image(
                            replay_buffer["gridmap_inputs"][idx], shape=image_shape, output_type="np", dtype="float32"
                        )
                    )
                    for idx in sample_indices
                ]
                next_grid_maps = [
                    torch.tensor(
                        decompress_image(
                            replay_buffer["next_gridmap_inputs"][idx],
                            shape=image_shape,
                            output_type="np",
                            dtype="float32",
                        )
                    )
                    for idx in sample_indices
                ]

            gridmap_batch = torch.stack(grid_maps).to(device)
            next_gridmap_batch = torch.stack(next_grid_maps).to(device)
        else:
            gridmap_batch = torch.stack([replay_buffer["gridmap_inputs"][index] for index in sample_indices]).to(device)
            next_gridmap_batch = torch.stack(
                [replay_buffer["next_gridmap_inputs"][index] for index in sample_indices]
            ).to(device)
    else:
        gridmap_batch = None
        next_gridmap_batch = None

    rollouts = {}
    for key in train_param["replay_buffer_keys"]:
        if key == "gridmap_inputs":
            rollouts["gridmap_inputs"] = gridmap_batch
        elif key == "next_gridmap_inputs":
            rollouts["next_gridmap_inputs"] = next_gridmap_batch
        else:
            rollouts[key] = [replay_buffer[key][index] for index in sample_indices]

    # stack batch data to tensors
    node_inputs_batch = torch.stack(rollouts["node_inputs"]).to(device)
    edge_inputs_batch = torch.stack(rollouts["edge_inputs"]).to(device)
    current_index_batch = torch.stack(rollouts["current_index"]).to(device)
    node_padding_mask_batch = torch.stack(rollouts["node_padding_mask"]).to(device)
    edge_padding_mask_batch = torch.stack(rollouts["curr_node_edge_padding_mask"]).to(device)
    edge_mask_batch = torch.stack(rollouts["edge_mask"]).to(device)
    action_batch = torch.stack(rollouts["action"]).to(device)
    reward_batch = torch.stack(rollouts["reward"]).to(device)
    done_batch = torch.stack(rollouts["done"]).to(device)
    next_node_inputs_batch = torch.stack(rollouts["next_node_inputs"]).to(device)
    next_edge_inputs_batch = torch.stack(rollouts["next_edge_inputs"]).to(device)
    next_current_index_batch = torch.stack(rollouts["next_current_index"]).to(device)
    next_node_padding_mask_batch = torch.stack(rollouts["next_node_padding_mask"]).to(device)
    next_edge_padding_mask_batch = torch.stack(rollouts["next_curr_node_edge_padding_mask"]).to(device)
    next_edge_mask_batch = torch.stack(rollouts["next_edge_mask"]).to(device)

    model_input = (
        node_inputs_batch,
        edge_inputs_batch,
        current_index_batch,
        node_padding_mask_batch,
        edge_padding_mask_batch,
        edge_mask_batch,
        gridmap_batch,
    )
    model_input_next = (
        next_node_inputs_batch,
        next_edge_inputs_batch,
        next_current_index_batch,
        next_node_padding_mask_batch,
        next_edge_padding_mask_batch,
        next_edge_mask_batch,
        next_gridmap_batch,
    )
    return model_input, model_input_next, action_batch, reward_batch, done_batch


class DictReplayBuffer:
    def __init__(self, max_size, keys, device="cpu", logger=None, img_compressed=False):
        self.max_size = max_size
        self.buffer = {key: [] for key in keys}
        self.device = device
        self.img_compressed = img_compressed
        if logger != None:
            self.logprint = logger.info
        else:
            self.logprint = print
        self.episode_lens_prefix_sum = PrefixSum(max_size)

    def add(self, episode_data):
        for key in episode_data.keys():
            self.buffer[key].extend(episode_data[key])
        buffer_size = len(self.buffer["done"])
        if buffer_size > self.max_size:
            self.logprint("Replay buffer overflow")
            for key in self.buffer.keys():
                self.buffer[key] = self.buffer[key][buffer_size - self.max_size :]
            buffer_size = len(self.buffer["done"])
        self.episode_lens_prefix_sum.add(len(episode_data["done"]))

    def sample(self, batch_size):
        return sample_batch(self.buffer, batch_size, self.episode_lens_prefix_sum, self.device)

    def __len__(self):
        return len(self.buffer["done"])


if __name__ == "__main__":
    if 1:  # test image compression
        img = Image.open("./tmp/map.png")
        if 0:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=img.format or "PNG")
            img_byte_data = img_byte_arr.getvalue()
            print(len(img_byte_data))
            compressed = compress_image(img, level=12)
            print(len(compressed))
            decompressed = decompress_image(compressed)
            print(decompressed.size)
            decompressed.save("tmp/decompressed.png")
            print(decompressed == img)
        if 1:
            img = np.array(img, dtype=np.uint8)
            compressed = compress_image(img, level=12)
            print(len(compressed))
            decompressed = decompress_image(compressed, output_type="np")
            print(decompressed.shape)
            decompressed = Image.fromarray(decompressed)
            decompressed.save("tmp/decompressed.png")
            print(decompressed == img)
    if 0:  # test batch compression
        batch_size = 128
        img = Image.open("tmp/map.png").convert("L")
        img_shape = img.size
        images = [np.array(img, dtype=np.uint8) for _ in range(batch_size)]
        compressed = compress_image_seq(images, verbose=True, level=16)
        decompressed = decompress_image_seq(compressed, img_shape, batch_size, dtype=np.int8, verbose=True)
        print(decompressed[0] == images[0])
    if 0:  # test prefix sum
        prefix_sum = PrefixSum(5)
        prefix_sum.add(2)
        print(f"ar: {prefix_sum.ar}")
        print(f"prefix_sum: {prefix_sum.prefix_sum}")
        range_idx = prefix_sum.get_range_idx(0)
        print(0, range_idx, prefix_sum.get_range_relative_idx(0, range_idx))

        prefix_sum.add(2)
        print(f"ar: {prefix_sum.ar}")
        print(f"prefix_sum: {prefix_sum.prefix_sum}")
        range_idx = prefix_sum.get_range_idx(1)
        print(1, range_idx, prefix_sum.get_range_relative_idx(1, range_idx))

        prefix_sum.add(3)
        prefix_sum.add(2)
        prefix_sum.add(1)
        print(f"ar: {prefix_sum.ar}")
        print(f"prefix_sum: {prefix_sum.prefix_sum}")
        for i in range(4, 9):
            range_idx = prefix_sum.get_range_idx(i)
            print(i, range_idx, prefix_sum.get_range_relative_idx(i, range_idx))

        prefix_sum.add(2)
        prefix_sum.add(2)
        print(f"new ar: {prefix_sum.ar}")
        print(f"new prefix_sum: {prefix_sum.prefix_sum}")
        for i in range(4, 9):
            range_idx = prefix_sum.get_range_idx(i)
            print(i, range_idx, prefix_sum.get_range_relative_idx(i, range_idx))
