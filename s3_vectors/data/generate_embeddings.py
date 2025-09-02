#!/usr/bin/env python3
"""
Robust embedding generator for the full Instacart dataset using Sentence Transformers.
Features:
- Progress tracking and resumable processing
- Local storage of embeddings
- S3 Vectors compatible format
- Error handling and retry logic
- Performance monitoring
"""

# Configuration Variables
INPUT_FILE = "instacart_sample_data.jsonl"
OUTPUT_DIR = "sentence_transformer_embeddings"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 200

import json
import time
import hashlib
import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import pickle
from datetime import datetime
import signal
import sys

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("❌ sentence-transformers not installed. Install with: pip install sentence-transformers")
    sys.exit(1)

class InstacartSentenceTransformerProcessor:
    def __init__(self, 
                 input_file: str = "instacart.jsonl",
                 output_dir: str = "sentence_transformer_embeddings",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 100):
        
        self.input_file = input_file
        self.output_dir = output_dir
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Progress tracking files
        self.progress_file = os.path.join(output_dir, "progress.json")
        self.embeddings_file = os.path.join(output_dir, "embeddings.jsonl")
        self.metadata_file = os.path.join(output_dir, "metadata.json")
        self.error_log_file = os.path.join(output_dir, "errors.log")
        
        # Initialize Sentence Transformer model
        try:
            print(f"📥 Loading Sentence Transformer model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.dimensions = self.model.get_sentence_embedding_dimension()
            print(f"✅ Model loaded successfully")
            print(f"📊 Embedding dimensions: {self.dimensions}")
        except Exception as e:
            print(f"❌ Failed to load Sentence Transformer model: {str(e)}")
            sys.exit(1)
        
        # Load or initialize progress
        self.progress = self.load_progress()
        
        # Set up graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print(f"🚀 Instacart Sentence Transformer Processor Initialized")
        print(f"   Input file: {input_file}")
        print(f"   Output directory: {output_dir}")
        print(f"   Model: {model_name}")
        print(f"   Dimensions: {self.dimensions}")
        print(f"   Batch size: {batch_size}")
    
    def signal_handler(self, signum, frame):
        """Handle graceful shutdown on Ctrl+C or SIGTERM."""
        print(f"\n🛑 Received signal {signum}. Saving progress and shutting down gracefully...")
        self.save_progress()
        print(f"💾 Progress saved. You can resume later by running the script again.")
        sys.exit(0)
    
    def load_progress(self) -> Dict:
        """Load progress from previous run."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                print(f"📊 Resuming from previous run:")
                print(f"   Processed: {progress.get('processed_count', 0)} items")
                print(f"   Last processed ID: {progress.get('last_processed_id', 'None')}")
                print(f"   Success rate: {progress.get('success_rate', 0):.1f}%")
                return progress
            except Exception as e:
                print(f"⚠️ Could not load progress file: {str(e)}")
        
        # Initialize new progress
        return {
            'processed_count': 0,
            'success_count': 0,
            'error_count': 0,
            'last_processed_id': None,
            'start_time': time.time(),
            'last_save_time': time.time(),
            'total_processing_time': 0,
            'success_rate': 0,
            'model_name': self.model_name,
            'dimensions': self.dimensions
        }
    
    def save_progress(self):
        """Save current progress to file."""
        self.progress['last_save_time'] = time.time()
        self.progress['success_rate'] = (self.progress['success_count'] / max(1, self.progress['processed_count'])) * 100
        self.progress['model_name'] = self.model_name
        self.progress['dimensions'] = self.dimensions
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save progress: {str(e)}")
    
    def log_error(self, product_id: str, error: str):
        """Log errors to file."""
        timestamp = datetime.now().isoformat()
        error_entry = f"[{timestamp}] Product {product_id}: {error}\n"
        
        try:
            with open(self.error_log_file, 'a') as f:
                f.write(error_entry)
        except Exception as e:
            print(f"⚠️ Could not write to error log: {str(e)}")
    
    def create_text_for_embedding(self, item: Dict[str, Any]) -> str:
        """Create text representation for embedding from product data."""
        title = item.get('title', '')
        categories = item.get('categories', [])
        
        # Handle different categories data types
        if isinstance(categories, list):
            categories_text = ', '.join(str(cat) for cat in categories)
        elif isinstance(categories, str):
            categories_text = categories
        else:
            categories_text = str(categories) if categories else ''
        
        # Combine title and categories for richer embedding
        text = f"{title}. Categories: {categories_text}"
        return text
    
    def generate_embeddings_batch(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generate embeddings for a batch of texts using Sentence Transformers."""
        try:
            embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False)
            return embeddings.tolist()  # Convert numpy array to list
            
        except Exception as e:
            raise Exception(f"Sentence Transformer error: {str(e)}")
    
    def prepare_s3_vectors_format(self, product: Dict[str, Any], embedding: List[float]) -> Dict:
        """Prepare data in S3 Vectors compatible format."""
        categories = product.get('categories', [])
        primary_category = categories[0] if isinstance(categories, list) and categories else "unknown"
        
        # Clean categories for S3 Vectors
        clean_categories = []
        if isinstance(categories, list):
            for cat in categories:
                if cat and str(cat).strip():
                    clean_categories.append(str(cat).strip())
        
        # Create text for metadata
        text = self.create_text_for_embedding(product)
        
        # Prepare metadata (S3 Vectors compatible)
        metadata = {
            "product_id": str(product.get('id', 'unknown')),
            "title": str(product.get('title', ''))[:500],  # Limit length
            "primary_category": str(primary_category),
            "source_text": str(text)[:1000],  # Limit length
            "dimensions": int(self.dimensions),
            "model_name": str(self.model_name)
        }
        
        # Only add categories array if it has items
        if clean_categories:
            metadata["all_categories"] = clean_categories
        
        return {
            "key": f"st_product_{product.get('id', 'unknown')}",  # Different prefix for sentence transformers
            "data": {"float32": embedding},
            "metadata": metadata,
            "product_id": product.get('id', 'unknown'),  # For tracking
            "embedding": embedding  # For local analysis
        }
    
    def count_total_items(self) -> int:
        """Count total items in the input file."""
        if hasattr(self, '_total_count'):
            return self._total_count
        
        print("📊 Counting total items in dataset...")
        count = 0
        try:
            with open(self.input_file, 'r') as f:
                for line in f:
                    if line.strip():
                        count += 1
        except Exception as e:
            print(f"⚠️ Could not count items: {str(e)}")
            return 0
        
        self._total_count = count
        print(f"📊 Total items in dataset: {count:,}")
        return count
    
    def should_skip_item(self, product_id: str) -> bool:
        """Check if item should be skipped based on progress."""
        if self.progress['last_processed_id'] is None:
            return False
        
        # Skip items until we reach the last processed ID
        return str(product_id) != str(self.progress['last_processed_id'])
    
    def process_dataset(self):
        """Process the entire dataset with progress tracking."""
        total_items = self.count_total_items()
        
        if total_items == 0:
            print(f"❌ No items found in {self.input_file}")
            return
        
        print(f"\n🚀 Starting processing...")
        print(f"   Total items: {total_items:,}")
        print(f"   Starting from item: {self.progress['processed_count'] + 1}")
        
        # Open files for writing
        embeddings_mode = 'a' if os.path.exists(self.embeddings_file) else 'w'
        
        start_time = time.time()
        batch_start_time = start_time
        items_in_batch = 0
        found_resume_point = self.progress['last_processed_id'] is None
        
        # Batch processing for sentence transformers
        current_batch_products = []
        current_batch_texts = []
        
        try:
            with open(self.input_file, 'r') as input_f, \
                 open(self.embeddings_file, embeddings_mode) as output_f:
                
                # Create progress bar
                pbar = tqdm(
                    total=total_items,
                    initial=self.progress['processed_count'],
                    desc="Processing products",
                    unit="items"
                )
                
                for line_num, line in enumerate(input_f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        product = json.loads(line)
                        product_id = product.get('id', f'line_{line_num}')
                        
                        # Skip items until we reach resume point
                        if not found_resume_point:
                            if str(product_id) == str(self.progress['last_processed_id']):
                                found_resume_point = True
                            continue
                        
                        # Add to current batch
                        text = self.create_text_for_embedding(product)
                        current_batch_products.append(product)
                        current_batch_texts.append(text)
                        
                        # Process batch when full or at end of file
                        if len(current_batch_products) >= 32 or line_num == total_items:
                            try:
                                # Generate embeddings for batch
                                embeddings = self.generate_embeddings_batch(current_batch_texts)
                                
                                if embeddings and len(embeddings) == len(current_batch_products):
                                    # Process each item in the batch
                                    for prod, emb in zip(current_batch_products, embeddings):
                                        # Prepare S3 Vectors format
                                        vector_data = self.prepare_s3_vectors_format(prod, emb)
                                        
                                        # Save to file
                                        output_f.write(json.dumps(vector_data) + '\n')
                                        
                                        self.progress['success_count'] += 1
                                        self.progress['processed_count'] += 1
                                        self.progress['last_processed_id'] = prod.get('id', f'line_{line_num}')
                                        items_in_batch += 1
                                        
                                        # Update progress bar
                                        pbar.update(1)
                                        pbar.set_postfix({
                                            'Success': f"{self.progress['success_count']:,}",
                                            'Errors': f"{self.progress['error_count']:,}",
                                            'Rate': f"{self.progress['success_count']/(time.time()-start_time):.1f}/s"
                                        })
                                else:
                                    raise Exception("Embedding batch size mismatch")
                            
                            except Exception as e:
                                # Handle batch errors
                                error_msg = str(e)
                                for prod in current_batch_products:
                                    prod_id = prod.get('id', 'unknown')
                                    self.log_error(prod_id, error_msg)
                                    self.progress['error_count'] += 1
                                    self.progress['processed_count'] += 1
                                    items_in_batch += 1
                                    pbar.update(1)
                                
                                # Print error for first few failures
                                if self.progress['error_count'] <= 5:
                                    print(f"\n⚠️ Error processing batch: {error_msg}")
                            
                            # Clear batch
                            current_batch_products = []
                            current_batch_texts = []
                            output_f.flush()  # Ensure data is written
                        
                        # Save progress periodically
                        if items_in_batch >= self.batch_size:
                            batch_time = time.time() - batch_start_time
                            items_per_second = items_in_batch / batch_time
                            
                            self.save_progress()
                            
                            print(f"\n📊 Batch completed: {items_in_batch} items in {batch_time:.1f}s ({items_per_second:.1f} items/s)")
                            
                            items_in_batch = 0
                            batch_start_time = time.time()
                        
                        # Small delay to avoid overwhelming the system
                        if len(current_batch_products) == 0:  # Only delay after processing a batch
                            time.sleep(0.01)
                    
                    except json.JSONDecodeError:
                        print(f"⚠️ Invalid JSON on line {line_num}")
                        continue
                
                pbar.close()
        
        except KeyboardInterrupt:
            print(f"\n🛑 Processing interrupted by user")
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}")
        finally:
            # Final save
            self.save_progress()
            self.print_final_summary()
    
    def print_final_summary(self):
        """Print final processing summary."""
        total_time = time.time() - self.progress['start_time']
        
        print(f"\n{'='*60}")
        print("📊 PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"🤖 Model: {self.model_name}")
        print(f"📊 Dimensions: {self.dimensions}")
        print(f"✅ Successfully processed: {self.progress['success_count']:,} items")
        print(f"❌ Errors: {self.progress['error_count']:,} items")
        print(f"📊 Total processed: {self.progress['processed_count']:,} items")
        print(f"📈 Success rate: {self.progress['success_rate']:.1f}%")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        
        if self.progress['success_count'] > 0:
            avg_time = total_time / self.progress['success_count']
            print(f"🚀 Average time per item: {avg_time:.2f} seconds")
            print(f"📈 Processing rate: {self.progress['success_count']/total_time:.1f} items/second")
        
        print(f"\n📁 Output files:")
        print(f"   Embeddings: {self.embeddings_file}")
        print(f"   Progress: {self.progress_file}")
        print(f"   Errors: {self.error_log_file}")
        
        if os.path.exists(self.embeddings_file):
            file_size = os.path.getsize(self.embeddings_file) / (1024 * 1024)  # MB
            print(f"   Embeddings file size: {file_size:.1f} MB")

def main():
    """Main function."""
    print("🚀 Instacart Dataset Sentence Transformer Embedding Generator")
    print("=" * 60)
    
    # Check if sentence-transformers is available
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("❌ sentence-transformers not installed")
        print("💡 Install with: pip install sentence-transformers")
        return
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        print(f"💡 Make sure {INPUT_FILE} is in the current directory")
        return
    
    # Initialize processor
    processor = InstacartSentenceTransformerProcessor(
        input_file=INPUT_FILE,
        output_dir=OUTPUT_DIR,
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE
    )
    
    # Process dataset
    processor.process_dataset()
    
    print(f"\n💡 To resume processing later, simply run this script again.")
    print(f"💡 To upload to S3 Vectors, use the sentence transformer upload script.")

if __name__ == "__main__":
    main()