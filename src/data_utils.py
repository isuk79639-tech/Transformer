import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pickle
import os
import re
from tqdm import tqdm
import numpy as np
import string
import xml.etree.ElementTree as ET

class Vocabulary:
    """Vocabulary class for token management"""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        
        self.pad_idx = 0
        self.unk_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3
        
        # Initialize with special tokens
        self.word2idx = {
            self.PAD_TOKEN: self.pad_idx,
            self.UNK_TOKEN: self.unk_idx,
            self.SOS_TOKEN: self.sos_idx,
            self.EOS_TOKEN: self.eos_idx
        }
        
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        
    def add_word(self, word):
        """Add word to vocabulary"""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.word_count[word] += 1
        
    def add_sentence(self, sentence):
        """Add all words in sentence to vocabulary"""
        for word in sentence.split():
            self.add_word(word)
    
    def build_vocab(self, sentences, min_freq=2):
        """Build vocabulary from sentences"""
        # Count word frequencies
        for sentence in sentences:
            for word in sentence.split():
                self.word_count[word] += 1
        
        # Add words that meet minimum frequency
        for word, count in self.word_count.items():
            if count >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def sentence_to_indices(self, sentence, max_length=None):
        """Convert sentence to indices"""
        words = sentence.split()
        indices = [self.sos_idx]
        
        for word in words:
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                indices.append(self.unk_idx)
        
        indices.append(self.eos_idx)
        
        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                indices.extend([self.pad_idx] * (max_length - len(indices)))
        
        return indices
    
    def indices_to_sentence(self, indices):
        """Convert indices to sentence"""
        words = []
        for idx in indices:
            if idx == self.eos_idx:
                break
            if idx in self.idx2word and idx not in [self.pad_idx, self.sos_idx]:
                words.append(self.idx2word[idx])
        return ' '.join(words)
    
    def __len__(self):
        return len(self.word2idx)
    
    def save(self, path):
        """Save vocabulary to file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_count': self.word_count
            }, f)
    
    def load(self, path):
        """Load vocabulary from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.word_count = data['word_count']

class TranslationDataset(Dataset):
    """Dataset for machine translation"""
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_length=100):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        
        # Convert to indices
        src_indices = self.src_vocab.sentence_to_indices(src_sentence, self.max_length)
        tgt_indices = self.tgt_vocab.sentence_to_indices(tgt_sentence, self.max_length)
        
        # Create target input (without EOS) and target output (without SOS)
        tgt_input = tgt_indices[:-1]
        tgt_output = tgt_indices[1:]
        
        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt_input': torch.tensor(tgt_input, dtype=torch.long),
            'tgt_output': torch.tensor(tgt_output, dtype=torch.long)
        }

class DataProcessor:
    """Data processor for IWSLT2017 dataset"""
    def __init__(self, data_dir='data', max_length=100, min_freq=2, use_real_data=False):
        self.data_dir = data_dir
        self.max_length = max_length
        self.min_freq = min_freq
        self.use_real_data = use_real_data  # Flag to control whether to use real IWSLT2017 data
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize tokenizer (simple whitespace tokenizer)
        self.tokenizer = self._simple_tokenizer
    
    def enable_real_data(self):
        """Enable loading real IWSLT2017 dataset"""
        self.use_real_data = True
        print("Real IWSLT2017 dataset enabled. Will load from 'en-de' directory.")
    
    def disable_real_data(self):
        """Disable real data loading, use dummy data instead"""
        self.use_real_data = False
        print("Using dummy dataset.")
    
    def _simple_tokenizer(self, text):
        """Simple tokenizer that splits on whitespace and punctuation"""
        # Remove punctuation and split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
        
    def preprocess_text(self, text):
        """Preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters except basic punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        return text.strip()
    
    def parse_tags_file(self, file_path):
        """Parse .tags format file from IWSLT2017 training data"""
        sentences = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and XML tags
                if not line or line.startswith('<'):
                    continue
                # Remove any remaining XML tags in the line
                line = re.sub(r'<[^>]+>', '', line)
                line = self.preprocess_text(line)
                if line:  # Only add non-empty sentences
                    sentences.append(line)
        return sentences
    
    def parse_xml_file(self, file_path):
        """Parse XML format file from IWSLT2017 dev/test data"""
        sentences = []
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Find all <seg> elements
            for seg in root.iter('seg'):
                text = seg.text
                if text:
                    text = self.preprocess_text(text)
                    if text:  # Only add non-empty sentences
                        sentences.append(text)
        except ET.ParseError as e:
            print(f"Warning: Error parsing XML file {file_path}: {e}")
            # Fallback: try to extract segments using regex
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract text from <seg> tags
                seg_pattern = r'<seg[^>]*>(.*?)</seg>'
                matches = re.findall(seg_pattern, content, re.DOTALL)
                for match in matches:
                    text = self.preprocess_text(match)
                    if text:
                        sentences.append(text)
        
        return sentences
    
    def load_iwslt_data(self, split='train', iwslt_dir='en-de'):
        # Default to dummy data unless explicitly enabled
        if not self.use_real_data:
            return self.create_dummy_data(split)
        
        # Try to load real IWSLT2017 data
        # Try to find IWSLT directory
        # First try relative to current working directory
        iwslt_path = iwslt_dir if os.path.exists(iwslt_dir) else None
        
        # If not found, try relative to this file's directory
        if iwslt_path is None:
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                potential_path = os.path.join(current_dir, iwslt_dir)
                if os.path.exists(potential_path):
                    iwslt_path = potential_path
            except:
                pass
        
        # If still not found, try in parent directory
        if iwslt_path is None:
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                potential_path = os.path.join(os.path.dirname(current_dir), iwslt_dir)
                if os.path.exists(potential_path):
                    iwslt_path = potential_path
            except:
                pass
        
        # Final fallback to dummy data if real data not found
        if iwslt_path is None or not os.path.exists(iwslt_path):
            print(f"Warning: IWSLT2017 directory not found at {iwslt_dir}. Using dummy data...")
            return self.create_dummy_data(split)
        
        if split == 'train':
            # Load training data from .tags files
            en_file = os.path.join(iwslt_path, 'train.tags.en-de.en')
            de_file = os.path.join(iwslt_path, 'train.tags.en-de.de')
            
            if not os.path.exists(en_file) or not os.path.exists(de_file):
                print(f"Warning: Training files not found. Using dummy data...")
                return self.create_dummy_data(split)
            
            print(f"Loading IWSLT2017 training data from {iwslt_path}...")
            src_sentences = self.parse_tags_file(en_file)
            tgt_sentences = self.parse_tags_file(de_file)
            
            # Align sentence pairs (they should be the same length)
            min_len = min(len(src_sentences), len(tgt_sentences))
            src_sentences = src_sentences[:min_len]
            tgt_sentences = tgt_sentences[:min_len]
            
            print(f"Loaded {len(src_sentences)} training sentence pairs")
            
        elif split == 'valid':
            # Load validation data from XML files (dev2010)
            en_file = os.path.join(iwslt_path, 'IWSLT17.TED.dev2010.en-de.en.xml')
            de_file = os.path.join(iwslt_path, 'IWSLT17.TED.dev2010.en-de.de.xml')
            
            if not os.path.exists(en_file) or not os.path.exists(de_file):
                print(f"Warning: Validation files not found. Using dummy data...")
                return self.create_dummy_data(split)
            
            print(f"Loading IWSLT2017 validation data from {iwslt_path}...")
            src_sentences = self.parse_xml_file(en_file)
            tgt_sentences = self.parse_xml_file(de_file)
            
            # Align sentence pairs
            min_len = min(len(src_sentences), len(tgt_sentences))
            src_sentences = src_sentences[:min_len]
            tgt_sentences = tgt_sentences[:min_len]
            
            print(f"Loaded {len(src_sentences)} validation sentence pairs")
            
        else:  # test
            # Load test data from XML files (use tst2010 as default)
            en_file = os.path.join(iwslt_path, 'IWSLT17.TED.tst2010.en-de.en.xml')
            de_file = os.path.join(iwslt_path, 'IWSLT17.TED.tst2010.en-de.de.xml')
            
            if not os.path.exists(en_file) or not os.path.exists(de_file):
                print(f"Warning: Test files not found. Using dummy data...")
                return self.create_dummy_data(split)
            
            print(f"Loading IWSLT2017 test data from {iwslt_path}...")
            src_sentences = self.parse_xml_file(en_file)
            tgt_sentences = self.parse_xml_file(de_file)
            
            # Align sentence pairs
            min_len = min(len(src_sentences), len(tgt_sentences))
            src_sentences = src_sentences[:min_len]
            tgt_sentences = tgt_sentences[:min_len]
            
            print(f"Loaded {len(src_sentences)} test sentence pairs")
        
        return src_sentences, tgt_sentences
    
    def create_dummy_data(self, split='train'):
        """Create dummy data for testing"""
        if split == 'train':
            size = 1000
        elif split == 'valid':
            size = 100
        else:
            size = 100
        
        # Simple English-German sentence pairs
        src_sentences = [
            "hello world",
            "how are you",
            "good morning",
            "thank you very much",
            "have a nice day",
            "see you later",
            "what is your name",
            "nice to meet you",
            "where are you from",
            "how old are you"
        ] * (size // 10)
        
        tgt_sentences = [
            "hallo welt",
            "wie geht es dir",
            "guten morgen",
            "vielen dank",
            "habe einen schönen tag",
            "bis später",
            "wie ist dein name",
            "freut mich dich kennenzulernen",
            "woher kommst du",
            "wie alt bist du"
        ] * (size // 10)
        
        return src_sentences, tgt_sentences
    
    def build_vocabularies(self, src_sentences, tgt_sentences):
        """Build source and target vocabularies"""
        print("Building vocabularies...")
        
        # Source vocabulary
        src_vocab = Vocabulary()
        src_vocab.build_vocab(src_sentences, self.min_freq)
        
        # Target vocabulary
        tgt_vocab = Vocabulary()
        tgt_vocab.build_vocab(tgt_sentences, self.min_freq)
        
        print(f"Source vocabulary size: {len(src_vocab)}")
        print(f"Target vocabulary size: {len(tgt_vocab)}")
        
        return src_vocab, tgt_vocab
    
    def create_data_loaders(self, batch_size=32, num_workers=0):
        """Create data loaders for train, validation, and test sets"""
        # Load data
        train_src, train_tgt = self.load_iwslt_data('train')
        valid_src, valid_tgt = self.load_iwslt_data('valid')
        test_src, test_tgt = self.load_iwslt_data('test')
        
        # Build vocabularies
        src_vocab, tgt_vocab = self.build_vocabularies(train_src, train_tgt)
        
        # Save vocabularies
        src_vocab.save(os.path.join(self.data_dir, 'src_vocab.pkl'))
        tgt_vocab.save(os.path.join(self.data_dir, 'tgt_vocab.pkl'))
        
        # Create datasets
        train_dataset = TranslationDataset(
            train_src, train_tgt, src_vocab, tgt_vocab, self.max_length
        )
        valid_dataset = TranslationDataset(
            valid_src, valid_tgt, src_vocab, tgt_vocab, self.max_length
        )
        test_dataset = TranslationDataset(
            test_src, test_tgt, src_vocab, tgt_vocab, self.max_length
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        return train_loader, valid_loader, test_loader, src_vocab, tgt_vocab

def collate_fn(batch):
    """Custom collate function for data loader"""
    src = torch.stack([item['src'] for item in batch])
    tgt_input = torch.stack([item['tgt_input'] for item in batch])
    tgt_output = torch.stack([item['tgt_output'] for item in batch])
    
    return {
        'src': src,
        'tgt_input': tgt_input,
        'tgt_output': tgt_output
    }

if __name__ == "__main__":
    # Test data processing
    processor = DataProcessor()
    train_loader, valid_loader, test_loader, src_vocab, tgt_vocab = processor.create_data_loaders()
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Valid batches: {len(valid_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    for batch in train_loader:
        print(f"Source shape: {batch['src'].shape}")
        print(f"Target input shape: {batch['tgt_input'].shape}")
        print(f"Target output shape: {batch['tgt_output'].shape}")
        break
