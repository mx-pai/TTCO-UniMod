"""
Long-sequence sampler for anti-drift training.
Samples consecutive frames (3-5 frames) instead of just 2 random frames.
"""
import random
import torch.utils.data
from lib.utils import TensorDict
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
import os.path


def no_processing(data):
    return data


class LongSeqTrackingSampler(torch.utils.data.Dataset):
    """
    Long-sequence sampler that samples 3-5 consecutive frames to train drift-resistant tracking.

    Training strategy:
    - Sample 1 template frame (t)
    - Sample 3-5 search frames (t+1, t+2, ..., t+k) consecutively
    - Train with accumulated tracking (search_i uses prediction from search_{i-1})
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames, num_template_frames=1, processing=no_processing,
                 seq_length=3, bert_model='bert-base-uncased', bert_path=None):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the template and first search frame
            num_search_frames - Not used in long-seq mode (always 1 search per forward)
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class
            seq_length - Number of consecutive frames to sample (3-5)
        """
        self.datasets = datasets
        self.seq_length = seq_length  # 3-5 consecutive frames

        if bert_path is not None and os.path.exists(bert_path):
            self.tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=True)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = 1  # Always 1 in long-seq mode
        self.num_template_frames = num_template_frames
        self.processing = processing
        self.max_seq_len = 40  # for NLP

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        """
        Returns a batch with:
        - 1 template frame
        - seq_length consecutive search frames
        - NLP description
        """
        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        is_video_dataset = dataset.is_video_sequence()

        # Sample a sequence
        seq_id = random.randint(0, dataset.get_num_sequences() - 1)

        # Sample frames
        seq_info_dict = dataset.get_sequence_info(seq_id)
        visible = seq_info_dict['visible']
        num_frames = len(visible)

        # We need at least (1 template + seq_length search) frames
        min_required_frames = 1 + self.seq_length

        if num_frames < min_required_frames:
            # Fallback: use all available frames
            actual_seq_len = max(1, num_frames - 1)
        else:
            actual_seq_len = self.seq_length

        # Sample template frame
        template_frame_ids = self._sample_visible_ids(visible, num_ids=1, max_id=num_frames - actual_seq_len)
        if template_frame_ids is None:
            template_frame_ids = [0]  # fallback

        template_frame_id = template_frame_ids[0]

        # Sample consecutive search frames starting from template_frame_id + gap
        gap = random.randint(1, min(self.max_gap, num_frames - template_frame_id - actual_seq_len))
        search_frame_ids = list(range(template_frame_id + gap, template_frame_id + gap + actual_seq_len))

        # Get frames and anno
        template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
        search_frames_list = []
        search_anno_list = []
        for sf_id in search_frame_ids:
            sf, sa, _ = dataset.get_frames(seq_id, [sf_id], seq_info_dict)
            search_frames_list.append(sf[0])
            search_anno_list.append(sa[0])

        # Get NLP
        nlp = seq_info_dict['nlp']
        nl_token_ids, nl_token_masks = self._extract_token_from_nlp(nlp, self.max_seq_len)

        # Prepare data dict
        data = TensorDict({
            'template_images': template_frames[0],
            'template_anno': template_anno[0],
            'search_images_seq': search_frames_list,  # List of consecutive frames
            'search_anno_seq': search_anno_list,
            'dataset': dataset.get_name(),
            'test_class': meta_obj_train.get('object_class_name'),
            'nl_token_ids': nl_token_ids,
            'nl_token_masks': nl_token_masks
        })

        return self.processing(data)

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def _extract_token_from_nlp(self, nlp, seq_length):
        """ Tokenize NLP """
        nlp_token = self.tokenizer.tokenize(nlp)
        if len(nlp_token) > seq_length - 2:
            nlp_token = nlp_token[0:(seq_length - 2)]

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in nlp_token:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        return torch.tensor(input_ids), torch.tensor(input_mask)

