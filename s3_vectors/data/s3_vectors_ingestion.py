#!/usr/bin/env python3
"""
Create S3 Vectors index for Sentence Transformers embeddings and upload all vectors.
Handles the 384-dimensional embeddings from all-MiniLM-L6-v2 model.
"""

# Configuration Variables
S3_VECTOR_BUCKET_NAME = "your-vector-bucket-name"
S3_VECTOR_INDEX_NAME = "your-vector-index-name"
EMBEDDINGS_DIR = "sentence_transformer_embeddings"
REGION_NAME = "us-east-1"
BATCH_SIZE = 500

import json
import time
import boto3
import os
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from datetime import datetime

class SentenceTransformerS3VectorsUploader:
    def __init__(self, 
                 embeddings_dir: str = "sentence_transformer_embeddings",
                 vector_bucket_name: str = "your-vector-bucket-name",
                 index_name: str = "your-vector-index-name",
                 region_name: str = "us-east-1",
                 batch_size: int = 500):
        
        self.embeddings_dir = embeddings_dir
        self.vector_bucket_name = vector_bucket_name
        self.index_name = index_name
        self.region_name = region_name
        self.batch_size = batch_size
        
        # Files
        self.embeddings_file = os.path.join(embeddings_dir, "embeddings.jsonl")
        self.upload_progress_file = os.path.join(embeddings_dir, "upload_progress.json")
        
        # Get dimensions from progress file
        self.dimensions = self.get_dimensions_from_progress()
        
        # Initialize S3 Vectors client
        try:
            self.s3vectors_client = boto3.client("s3vectors", region_name=region_name)
            print(f"✅ S3 Vectors client initialized in {region_name}")
        except Exception as e:
            print(f"❌ Failed to initialize S3 Vectors client: {str(e)}")
            raise
        
        # Load upload progress
        self.upload_progress = self.load_upload_progress()
        
        print(f"🚀 Sentence Transformer S3 Vectors Uploader Initialized")
        print(f"   Embeddings file: {self.embeddings_file}")
        print(f"   Vector bucket: {vector_bucket_name}")
        print(f"   Index: {index_name}")
        print(f"   Dimensions: {self.dimensions}")
        print(f"   Batch size: {batch_size}")
    
    def get_dimensions_from_progress(self) -> int:
        """Get embedding dimensions from the progress file."""
        progress_file = os.path.join(self.embeddings_dir, "progress.json")
        
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    return progress.get('dimensions', 384)  # Default to 384 for all-MiniLM-L6-v2
            except Exception as e:
                print(f"⚠️ Could not read dimensions from progress file: {str(e)}")
        
        return 384  # Default for all-MiniLM-L6-v2
    
    def load_upload_progress(self) -> Dict:
        """Load upload progress from previous run."""
        if os.path.exists(self.upload_progress_file):
            try:
                with open(self.upload_progress_file, 'r') as f:
                    progress = json.load(f)
                print(f"📊 Resuming upload from previous run:")
                print(f"   Uploaded: {progress.get('uploaded_count', 0)} vectors")
                print(f"   Last batch: {progress.get('last_batch', 0)}")
                return progress
            except Exception as e:
                print(f"⚠️ Could not load upload progress: {str(e)}")
        
        return {
            'uploaded_count': 0,
            'last_batch': 0,
            'start_time': time.time(),
            'error_count': 0,
            'index_created': False
        }
    
    def save_upload_progress(self):
        """Save upload progress."""
        try:
            with open(self.upload_progress_file, 'w') as f:
                json.dump(self.upload_progress, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save upload progress: {str(e)}")
    
    def list_vector_buckets(self) -> List[Dict[str, str]]:
        """List available S3 Vector buckets."""
        try:
            response = self.s3vectors_client.list_vector_buckets()
            buckets = []
            for bucket in response.get('vectorBuckets', []):
                buckets.append({
                    'name': bucket['vectorBucketName'],
                    'arn': bucket['vectorBucketArn'],
                    'creation_time': bucket['creationTime']
                })
            return buckets
        except Exception as e:
            print(f"❌ Error listing S3 Vector buckets: {str(e)}")
            return []
    
    def create_vector_bucket(self, bucket_name: str) -> bool:
        """Create a new S3 Vector bucket."""
        try:
            response = self.s3vectors_client.create_vector_bucket(
                vectorBucketName=bucket_name
            )
            print(f"✅ S3 Vector bucket '{bucket_name}' created successfully")
            return True
        except Exception as e:
            error_msg = str(e)
            if "AlreadyExistsException" in error_msg:
                print(f"✅ S3 Vector bucket '{bucket_name}' already exists")
                return True
            else:
                print(f"❌ Error creating S3 Vector bucket: {error_msg}")
                return False
    
    def list_vector_indexes(self) -> List[Dict[str, str]]:
        """List existing S3 Vector indexes in the bucket."""
        try:
            response = self.s3vectors_client.list_indexes(vectorBucketName=self.vector_bucket_name)
            indexes = []
            for index in response.get('indexes', []):
                indexes.append({
                    'name': index['indexName'],
                    'arn': index['indexArn'],
                    'bucket_name': index['vectorBucketName'],
                    'creation_time': index['creationTime']
                })
            return indexes
        except Exception as e:
            print(f"❌ Error listing S3 Vector indexes: {str(e)}")
            return []
    
    def create_index(self) -> bool:
        """Create the S3 Vectors index for sentence transformers."""
        if self.upload_progress.get('index_created', False):
            print(f"✅ Index '{self.index_name}' already created")
            return True
        
        print(f"🔧 Creating S3 Vectors index '{self.index_name}'...")
        print(f"   Dimensions: {self.dimensions}")
        print(f"   Distance metric: cosine")
        
        try:
            response = self.s3vectors_client.create_index(
                vectorBucketName=self.vector_bucket_name,
                indexName=self.index_name,
                dataType='float32',
                dimension=self.dimensions,
                distanceMetric='cosine',  # Best for text embeddings
                metadataConfiguration={
                    'nonFilterableMetadataKeys': [
                        'source_text',  # Large text fields don't need filtering
                        'dimensions',   # Numeric metadata that doesn't need filtering
                        'model_name'    # Model name doesn't need filtering
                    ]
                }
            )
            
            print("✅ Index created successfully")
            self.upload_progress['index_created'] = True
            self.save_upload_progress()
            
            # Wait for index to be ready
            print("⏳ Waiting for index to be ready...")
            time.sleep(5)
            
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "AlreadyExistsException" in error_msg:
                print("✅ Index already exists")
                self.upload_progress['index_created'] = True
                self.save_upload_progress()
                return True
            else:
                print(f"❌ Error creating index: {error_msg}")
                return False
    
    def count_embeddings(self) -> int:
        """Count total embeddings available for upload."""
        if not os.path.exists(self.embeddings_file):
            return 0
        
        count = 0
        try:
            with open(self.embeddings_file, 'r') as f:
                for line in f:
                    if line.strip():
                        count += 1
        except Exception as e:
            print(f"⚠️ Could not count embeddings: {str(e)}")
        
        return count
    
    def load_embeddings_batch(self, start_line: int, batch_size: int) -> List[Dict]:
        """Load a batch of embeddings starting from a specific line."""
        vectors = []
        
        try:
            with open(self.embeddings_file, 'r') as f:
                # Skip to start line
                for i in range(start_line):
                    next(f, None)
                
                # Read batch
                for i in range(batch_size):
                    line = next(f, None)
                    if line is None:
                        break
                    
                    if line.strip():
                        try:
                            vector_data = json.loads(line)
                            # Extract S3 Vectors format
                            s3_vector = {
                                "key": vector_data["key"],
                                "data": vector_data["data"],
                                "metadata": vector_data["metadata"]
                            }
                            vectors.append(s3_vector)
                        except json.JSONDecodeError:
                            continue
        
        except Exception as e:
            print(f"❌ Error loading batch: {str(e)}")
        
        return vectors
    
    def upload_batch(self, vectors: List[Dict], batch_num: int) -> bool:
        """Upload a batch of vectors to S3 Vectors."""
        try:
            response = self.s3vectors_client.put_vectors(
                vectorBucketName=self.vector_bucket_name,
                indexName=self.index_name,
                vectors=vectors
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Error uploading batch {batch_num}: {str(e)}")
            self.upload_progress['error_count'] += 1
            return False
    
    def upload_all_embeddings(self):
        """Upload all embeddings to S3 Vectors in batches."""
        total_embeddings = self.count_embeddings()
        
        if total_embeddings == 0:
            print(f"❌ No embeddings found in {self.embeddings_file}")
            return
        
        print(f"\n🚀 Starting S3 Vectors upload...")
        print(f"   Total embeddings: {total_embeddings:,}")
        print(f"   Batch size: {self.batch_size}")
        
        # Calculate batches
        total_batches = (total_embeddings + self.batch_size - 1) // self.batch_size
        start_batch = self.upload_progress['last_batch']
        
        print(f"   Total batches: {total_batches}")
        
        if start_batch > 0:
            print(f"   Resuming from batch: {start_batch + 1}")
        
        start_time = time.time()
        
        # Process all remaining batches
        with tqdm(total=total_batches - start_batch, desc="Uploading batches", unit="batch") as pbar:
            
            for batch_num in range(start_batch, total_batches):
                start_line = batch_num * self.batch_size
                
                # Load batch
                vectors = self.load_embeddings_batch(start_line, self.batch_size)
                
                if not vectors:
                    print(f"⚠️ No vectors in batch {batch_num + 1}")
                    continue
                
                # Upload batch
                success = self.upload_batch(vectors, batch_num + 1)
                
                if success:
                    self.upload_progress['uploaded_count'] += len(vectors)
                    self.upload_progress['last_batch'] = batch_num + 1
                    
                    pbar.set_postfix({
                        'Uploaded': f"{self.upload_progress['uploaded_count']:,}",
                        'Errors': f"{self.upload_progress['error_count']:,}",
                        'Batch': f"{batch_num + 1}/{total_batches}"
                    })
                else:
                    print(f"❌ Failed to upload batch {batch_num + 1}")
                    # Continue with next batch even if one fails
                
                pbar.update(1)
                
                # Save progress every 10 batches
                if (batch_num + 1) % 10 == 0:
                    self.save_upload_progress()
                
                # Small delay between batches to avoid rate limiting
                time.sleep(0.01)
        
        # Final save
        self.save_upload_progress()
        
        # Summary
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("📊 UPLOAD SUMMARY")
        print(f"{'='*60}")
        print(f"🤖 Model: Sentence Transformers (all-MiniLM-L6-v2)")
        print(f"📊 Dimensions: {self.dimensions}")
        print(f"🗂️ Index: {self.index_name}")
        print(f"✅ Uploaded: {self.upload_progress['uploaded_count']:,} vectors")
        print(f"❌ Errors: {self.upload_progress['error_count']:,} batches")
        print(f"⏱️  Time: {total_time/60:.1f} minutes")
        
        if self.upload_progress['uploaded_count'] > 0:
            rate = self.upload_progress['uploaded_count'] / total_time
            print(f"🚀 Upload rate: {rate:.1f} vectors/second")
        
        if self.upload_progress['uploaded_count'] >= total_embeddings:
            print(f"\n🎉 All sentence transformer embeddings uploaded successfully!")
        else:
            remaining = total_embeddings - self.upload_progress['uploaded_count']
            print(f"\n⚠️ Upload incomplete: {remaining:,} vectors remaining")
            print(f"💡 Run script again to retry failed uploads")

def select_or_create_bucket(uploader: SentenceTransformerS3VectorsUploader) -> Tuple[str, bool]:
    """Interactive S3 Vector bucket selection or creation."""
    print("\n🪣 S3 VECTOR BUCKET SELECTION")
    print("=" * 35)
    
    # List existing vector buckets
    buckets = uploader.list_vector_buckets()
    
    if buckets:
        print("📋 Available S3 Vector buckets:")
        for i, bucket in enumerate(buckets, 1):
            creation_date = bucket['creation_time'].strftime('%Y-%m-%d %H:%M:%S')
            print(f"   {i}. {bucket['name']} (created: {creation_date})")
        
        print(f"   {len(buckets) + 1}. Create new vector bucket")
        
        while True:
            try:
                choice = input(f"\nSelect vector bucket (1-{len(buckets) + 1}): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(buckets):
                    selected_bucket = buckets[choice_num - 1]['name']
                    print(f"✅ Selected existing vector bucket: {selected_bucket}")
                    return selected_bucket, False
                elif choice_num == len(buckets) + 1:
                    break
                else:
                    print(f"❌ Invalid choice. Please enter 1-{len(buckets) + 1}")
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
    else:
        print("📋 No existing S3 Vector buckets found.")
    
    # Create new vector bucket
    while True:
        bucket_name = input("\n🆕 Enter new vector bucket name: ").strip().lower()
        
        if not bucket_name:
            print("❌ Vector bucket name cannot be empty")
            continue
        
        # Basic bucket name validation
        if not bucket_name.replace('-', '').replace('_', '').isalnum():
            print("❌ Vector bucket name can only contain letters, numbers, hyphens, and underscores")
            continue
        
        if len(bucket_name) < 3 or len(bucket_name) > 63:
            print("❌ Vector bucket name must be between 3 and 63 characters")
            continue
        
        confirm = input(f"Create vector bucket '{bucket_name}'? (y/n): ").strip().lower()
        if confirm == 'y':
            if uploader.create_vector_bucket(bucket_name):
                return bucket_name, True
            else:
                print("❌ Failed to create vector bucket. Try a different name.")
        else:
            print("❌ Vector bucket creation cancelled")

def select_or_create_index(uploader: SentenceTransformerS3VectorsUploader) -> Tuple[str, bool]:
    """Interactive S3 Vector index selection or creation."""
    print("\n📊 S3 VECTOR INDEX SELECTION")
    print("=" * 35)
    
    # List existing indexes
    indexes = uploader.list_vector_indexes()
    
    if indexes:
        print(f"📋 Available S3 Vector indexes in bucket '{uploader.vector_bucket_name}':")
        for i, index in enumerate(indexes, 1):
            creation_date = index['creation_time'].strftime('%Y-%m-%d %H:%M:%S')
            print(f"   {i}. {index['name']} (created: {creation_date})")
        
        print(f"   {len(indexes) + 1}. Create new index")
        
        while True:
            try:
                choice = input(f"\nSelect index (1-{len(indexes) + 1}): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(indexes):
                    selected_index = indexes[choice_num - 1]['name']
                    print(f"✅ Selected existing index: {selected_index}")
                    return selected_index, False
                elif choice_num == len(indexes) + 1:
                    break
                else:
                    print(f"❌ Invalid choice. Please enter 1-{len(indexes) + 1}")
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
    else:
        print(f"📋 No existing S3 Vector indexes found in bucket '{uploader.vector_bucket_name}'.")
    
    # Create new index
    while True:
        index_name = input("\n🆕 Enter new index name: ").strip()
        
        if not index_name:
            print("❌ Index name cannot be empty")
            continue
        
        # Basic index name validation
        if not index_name.replace('-', '').replace('_', '').isalnum():
            print("❌ Index name can only contain letters, numbers, hyphens, and underscores")
            continue
        
        confirm = input(f"Create index '{index_name}' in bucket '{uploader.vector_bucket_name}'? (y/n): ").strip().lower()
        if confirm == 'y':
            return index_name, True
        else:
            print("❌ Index creation cancelled")

def main():
    """Main function."""
    start_time = time.time()
    print("🚀 Sentence Transformer S3 Vectors Uploader")
    print("=" * 50)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if embeddings exist
    embeddings_file = f"{EMBEDDINGS_DIR}/embeddings.jsonl"
    if not os.path.exists(embeddings_file):
        print(f"❌ Embeddings file not found: {embeddings_file}")
        print("💡 Run generate_embeddings.py first to generate embeddings")
        return
    
    # Initialize temporary uploader for bucket/index selection
    temp_uploader = SentenceTransformerS3VectorsUploader(
        embeddings_dir=EMBEDDINGS_DIR,
        vector_bucket_name=S3_VECTOR_BUCKET_NAME,  # Default, will be updated
        index_name=S3_VECTOR_INDEX_NAME,  # Default, will be updated
        region_name=REGION_NAME,
        batch_size=BATCH_SIZE
    )
    
    # Interactive vector bucket selection
    try:
        selected_bucket, bucket_created = select_or_create_bucket(temp_uploader)
        if bucket_created:
            print(f"⏳ Waiting for vector bucket to be ready...")
            time.sleep(3)
    except Exception as e:
        print(f"❌ Error with vector bucket selection: {str(e)}")
        return
    
    # Update bucket name for index operations
    temp_uploader.vector_bucket_name = selected_bucket
    
    # Interactive index selection
    try:
        selected_index, index_will_be_created = select_or_create_index(temp_uploader)
    except Exception as e:
        print(f"❌ Error with index selection: {str(e)}")
        return
    
    # Initialize final uploader with selected vector bucket and index
    print(f"\n🔧 Initializing uploader with:")
    print(f"   Vector Bucket: {selected_bucket}")
    print(f"   Vector Index: {selected_index}")
    
    uploader = SentenceTransformerS3VectorsUploader(
        embeddings_dir=EMBEDDINGS_DIR,
        vector_bucket_name=selected_bucket,
        index_name=selected_index,
        region_name=REGION_NAME,
        batch_size=BATCH_SIZE
    )
    
    # Create index if needed
    index_creation_start = time.time()
    if index_will_be_created or not uploader.upload_progress.get('index_created', False):
        print(f"\n🔧 Creating/verifying index...")
        if not uploader.create_index():
            print("❌ Failed to create index. Cannot proceed with upload.")
            return
        index_creation_time = time.time() - index_creation_start
        print(f"⏱️ Index creation/verification took: {index_creation_time:.2f} seconds")
    else:
        print(f"\n✅ Using existing index: {selected_index}")
    
    # Check if upload is already complete
    total_embeddings = uploader.count_embeddings()
    if uploader.upload_progress['uploaded_count'] >= total_embeddings:
        print(f"\n⚠️ Upload appears to be complete ({uploader.upload_progress['uploaded_count']} vectors)")
        print(f"💡 If you cleared the S3 Vectors index, you need to reset upload progress")
        
        reset = input("Reset upload progress and start over? (y/n): ").strip().lower()
        if reset == 'y':
            # Reset upload progress but keep index_created flag
            uploader.upload_progress = {
                'uploaded_count': 0,
                'last_batch': 0,
                'start_time': time.time(),
                'error_count': 0,
                'index_created': True  # Keep this flag
            }
            uploader.save_upload_progress()
            print("✅ Upload progress reset")
        else:
            print("❌ Upload cancelled")
            return
    
    # Upload all embeddings
    upload_start_time = time.time()
    uploader.upload_all_embeddings()
    upload_time = time.time() - upload_start_time
    
    # Final timing summary
    total_time = time.time() - start_time
    print(f"\n⏰ TIMING SUMMARY")
    print(f"=" * 20)
    print(f"🕐 Total execution time: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
    print(f"📤 Upload time: {upload_time/60:.2f} minutes ({upload_time:.2f} seconds)")
    print(f"🏁 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if uploader.upload_progress['uploaded_count'] > 0:
        vectors_per_minute = uploader.upload_progress['uploaded_count'] / (upload_time / 60)
        print(f"📊 Average upload rate: {vectors_per_minute:.0f} vectors/minute")

if __name__ == "__main__":
    main()