#!/usr/bin/env python3
"""
Streamlit UI comparing S3 Vectors vs OpenSearch Serverless Query.
Side-by-side comparison of both search methods using the same embeddings.
"""

# Configuration Variables
# S3 Vectors Configuration
S3_VECTOR_BUCKET_NAME = "your-vector-bucket-name"
S3_VECTOR_INDEX_NAME = "your-vector-index-name"
S3_VECTOR_REGION = "us-east-1"

# OpenSearch Configuration
OPENSEARCH_HOST = "your-opensearch-host.us-east-1.aoss.amazonaws.com"
OPENSEARCH_INDEX_NAME = "your-opensearch-index-name"
OPENSEARCH_REGION = "us-east-1"

# Model Configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

import streamlit as st
import json
import time
import boto3
from typing import List, Dict, Any, Optional
import pandas as pd
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth


# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class S3VectorsSearch:
    """S3 Vectors search implementation."""
    def __init__(self, 
                 vector_bucket_name: str = S3_VECTOR_BUCKET_NAME,
                 index_name: str = S3_VECTOR_INDEX_NAME,
                 region_name: str = S3_VECTOR_REGION,
                 model_name: str = MODEL_NAME):
        
        self.vector_bucket_name = vector_bucket_name
        self.index_name = index_name
        self.region_name = region_name
        self.model_name = model_name
        
        # Initialize S3 Vectors client
        try:
            self.s3vectors_client = boto3.client("s3vectors", region_name=region_name)
            self.s3_initialized = True
        except Exception as e:
            st.error(f"❌ Failed to initialize S3 Vectors client: {str(e)}")
            self.s3_initialized = False
        
        # Initialize Sentence Transformer model
        self.model = None
        self.model_initialized = False
    
    def load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
                self.model_initialized = True
                return True
            except Exception as e:
                st.error(f"❌ Failed to load S3 Vectors model: {str(e)}")
                self.model_initialized = False
                return False
        return True
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query."""
        if not self.model_initialized:
            return []
        
        try:
            embedding = self.model.encode([query.strip()])[0]
            return embedding.tolist()
        except Exception as e:
            st.error(f"❌ S3 Vectors embedding error: {str(e)}")
            return []
    
    def search_products(self, query: str, top_k: int = 5, query_embedding: List[float] = None, category_filter: str = None) -> List[Dict]:
        """Search for products using S3 Vectors."""
        
        if not self.s3_initialized or not self.model_initialized:
            return []
        
        # Generate query embedding if not provided
        if query_embedding is None:
            query_embedding = self.generate_query_embedding(query)
        
        if not query_embedding:
            return []
        
        try:
            # Prepare query parameters
            query_params = {
                "vectorBucketName": self.vector_bucket_name,
                "indexName": self.index_name,
                "queryVector": {"float32": query_embedding},
                "topK": top_k,
                "returnDistance": True,
                "returnMetadata": True
            }
            
            # Add category filter if specified
            if category_filter and category_filter != "All Categories":
                query_params["filter"] = {"all_categories": {"$in": [category_filter]}}
            
            # Execute query
            start_time = time.time()
            response = self.s3vectors_client.query_vectors(**query_params)
            search_time = time.time() - start_time
            
            results = response.get('vectors', [])
            
            # Process and format results
            formatted_results = []
            for i, result in enumerate(results):
                metadata = result.get('metadata', {})
                distance = result.get('distance', 1.0)
                
                formatted_result = {
                    'rank': i + 1,
                    'title': metadata.get('title', ''),
                    'distance': distance,
                    'primary_category': metadata.get('primary_category', ''),
                    'all_categories': metadata.get('all_categories', []),
                    'product_id': metadata.get('product_id', ''),
                    'source_text': metadata.get('source_text', ''),
                    'search_time': search_time,
                    'method': 'S3 Vectors'
                }
                
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            st.error(f"❌ S3 Vectors search error: {str(e)}")
            return []

class OpenSearchDirectQuery:
    """OpenSearch direct query implementation."""
    def __init__(self, 
                 host: str = OPENSEARCH_HOST,
                 index_name: str = OPENSEARCH_INDEX_NAME,
                 region_name: str = OPENSEARCH_REGION,
                 model_name: str = MODEL_NAME):
        
        self.host = host
        self.index_name = index_name
        self.region_name = region_name
        self.model_name = model_name
        
        # Initialize AWS credentials for signing requests
        try:
            credentials = boto3.Session().get_credentials()
            self.awsauth = AWS4Auth(
                credentials.access_key,
                credentials.secret_key,
                region_name,
                'aoss',
                session_token=credentials.token
            )
            
            # Initialize OpenSearch client
            self.client = OpenSearch(
                hosts=[{'host': host, 'port': 443}],
                http_auth=self.awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                timeout=300
            )
            self.opensearch_initialized = True
        except Exception as e:
            st.error(f"❌ Failed to initialize OpenSearch client: {str(e)}")
            self.opensearch_initialized = False
        
        # Initialize Sentence Transformer model
        self.model = None
        self.model_initialized = False
    
    def load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
                self.model_initialized = True
                return True
            except Exception as e:
                st.error(f"❌ Failed to load OpenSearch model: {str(e)}")
                self.model_initialized = False
                return False
        return True
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query."""
        if not self.model_initialized:
            return []
        
        try:
            embedding = self.model.encode([query.strip()])[0]
            return embedding.tolist()
        except Exception as e:
            st.error(f"❌ OpenSearch embedding error: {str(e)}")
            return []
    
    def search_products(self, query: str, top_k: int = 5, query_embedding: List[float] = None, category_filter: str = None) -> List[Dict]:
        """Search for products using OpenSearch direct query."""
        
        if not self.opensearch_initialized or not self.model_initialized:
            return []
        
        # Generate query embedding if not provided
        if query_embedding is None:
            query_embedding = self.generate_query_embedding(query)
        
        if not query_embedding:
            return []
        
        # Prepare OpenSearch query with pre-filtering
        if category_filter and category_filter != "All Categories":
            # Use pre-filter with KNN - more efficient
            search_body = {
                "size": top_k,
                "query": {
                    "knn": {
                        "vector_record": {
                            "vector": query_embedding,
                            "k": top_k,
                            "filter": {
                                "term": {
                                    "metadata.all_categories.keyword": category_filter
                                }
                            }
                        }
                    }
                },
                "_source": {
                    "includes": ["metadata", "key"]
                }
            }
        else:
            # Standard KNN query without filter
            search_body = {
                "size": top_k,
                "query": {
                    "knn": {
                        "vector_record": {
                            "vector": query_embedding,
                            "k": top_k
                        }
                    }
                },
                "_source": {
                    "includes": ["metadata", "key"]
                }
            }
        
        try:
            start_time = time.time()
            
            # Execute search using opensearch-py client
            response = self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            search_time = time.time() - start_time
            
            # Parse response
            hits = response.get('hits', {}).get('hits', [])
            
            # Get the actual query execution time from OpenSearch (in milliseconds)
            opensearch_took_ms = response.get('took', 0)
            opensearch_took_seconds = opensearch_took_ms / 1000.0
            
            # Format results and limit to requested top_k
            formatted_results = []
            for i, hit in enumerate(hits[:top_k]):  # Limit to top_k results
                source = hit.get('_source', {})
                metadata = source.get('metadata', {})
                score = hit.get('_score', 0)
                
                # Convert score to distance
                distance = max(0, 1 - score) if score > 0 else 1.0
                
                formatted_result = {
                    'rank': i + 1,
                    'title': metadata.get('title', ''),
                    'distance': distance,
                    'primary_category': metadata.get('primary_category', ''),
                    'all_categories': metadata.get('all_categories', []),
                    'product_id': metadata.get('product_id', ''),
                    'source_text': metadata.get('source_text', ''),
                    'key': source.get('key', ''),
                    'search_time': search_time,  # Round-trip time
                    'opensearch_took': opensearch_took_seconds,  # Server-side execution time
                    'opensearch_took_ms': opensearch_took_ms,  # Server-side execution time in ms
                    'method': 'OpenSearch Direct'
                }
                
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            st.error(f"❌ OpenSearch search error: {str(e)}")
            return []



def generate_shared_embedding(search_engine, query: str) -> List[float]:
    """Generate embedding once using either search engine."""
    return search_engine.generate_query_embedding(query)

@st.cache_resource
def initialize_search_engines():
    """Initialize both search engines with caching."""
    s3_search = S3VectorsSearch(
        vector_bucket_name=S3_VECTOR_BUCKET_NAME,
        index_name=S3_VECTOR_INDEX_NAME,
        region_name=S3_VECTOR_REGION,
        model_name=MODEL_NAME
    )
    
    opensearch_search = OpenSearchDirectQuery(
        host=OPENSEARCH_HOST,
        index_name=OPENSEARCH_INDEX_NAME,
        region_name=OPENSEARCH_REGION,
        model_name=MODEL_NAME
    )
    
    return s3_search, opensearch_search

def display_comparison_results(s3_results: List[Dict], opensearch_results: List[Dict]):
    """Display side-by-side comparison of search results."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔵 S3 Vectors Results")
        if s3_results:
            # Create DataFrame for S3 results
            s3_df_data = []
            for result in s3_results:
                s3_df_data.append({
                    'Rank': result['rank'],
                    'Product': result['title'],
                    'Distance': f"{result['distance']:.4f}",
                    'Category': result['primary_category'],
                    'Product ID': result['product_id']
                })
            
            s3_df = pd.DataFrame(s3_df_data)
            st.dataframe(s3_df, use_container_width=True, hide_index=True)
        else:
            st.warning("❌ No S3 Vectors results")
    
    with col2:
        st.subheader("🟠 OpenSearch Direct Results")
        if opensearch_results:
            # Create DataFrame for OpenSearch results
            os_df_data = []
            for result in opensearch_results:
                os_df_data.append({
                    'Rank': result['rank'],
                    'Product': result['title'],
                    'Distance': f"{result['distance']:.4f}",
                    'Category': result['primary_category'],
                    'Product ID': result['product_id']
                })
            
            os_df = pd.DataFrame(os_df_data)
            st.dataframe(os_df, use_container_width=True, hide_index=True)
        else:
            st.warning("❌ No OpenSearch results")

def analyze_differences(s3_results: List[Dict], opensearch_results: List[Dict]):
    """Analyze and display differences between the two search methods."""
    
    if not s3_results or not opensearch_results:
        return
    
    st.subheader("🔍 Performance Comparison")
    
    # Performance comparison
    s3_time = s3_results[0]['search_time']
    os_time = opensearch_results[0]['search_time']
    os_server_time = opensearch_results[0].get('opensearch_took', 0)
    os_server_time_ms = opensearch_results[0].get('opensearch_took_ms', 0)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate network overhead from OpenSearch
    network_overhead = os_time - os_server_time if os_time > os_server_time else 0
    
    # Calculate estimated S3 Vectors server-side time
    s3_estimated_server_time = max(0, s3_time - network_overhead)
    s3_estimated_server_time_ms = s3_estimated_server_time * 1000
    
    # Determine which server-side time is faster
    s3_server_faster = s3_estimated_server_time < os_server_time
    os_server_faster = os_server_time < s3_estimated_server_time
    
    with col1:
        st.metric("S3 Vectors Round-trip", f"{s3_time:.3f}s")
    
    with col2:
        if s3_server_faster:
            st.metric("🏆 S3 Vectors Server-side (calculated)", f"{s3_estimated_server_time_ms:.0f}ms", f"-{(os_server_time - s3_estimated_server_time)*1000:.0f}ms faster")
        else:
            st.metric("S3 Vectors Server-side (calculated)", f"{s3_estimated_server_time_ms:.0f}ms")
    
    with col3:
        st.metric("OpenSearch Round-trip", f"{os_time:.3f}s")
    
    with col4:
        if os_server_faster:
            st.metric("🏆 OpenSearch Server-side", f"{os_server_time_ms}ms", f"-{(s3_estimated_server_time - os_server_time)*1000:.0f}ms faster")
        else:
            st.metric("OpenSearch Server-side", f"{os_server_time_ms}ms")
    
    # Additional breakdown
    if os_time > os_server_time:
        network_overhead = os_time - os_server_time
        st.info(f"📡 **Network overhead:** {network_overhead:.3f}s ({network_overhead*1000:.1f}ms) - "
                f"This is the difference between round-trip time and server execution time")



def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Search Method Comparison",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ S3 Vectors vs OpenSearch Direct Query Comparison")
    st.markdown("Compare search results between S3 Vectors managed service and direct OpenSearch Serverless queries")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model info
        st.subheader("🤖 Model Information")
        st.write("**Model:** all-MiniLM-L6-v2")
        st.write("**Dimensions:** 384")
        st.write("**Distance Metric:** Cosine")
        
        st.subheader("🔵 S3 Vectors")
        st.write(f"**Bucket:** {S3_VECTOR_BUCKET_NAME}")
        st.write(f"**Index:** {S3_VECTOR_INDEX_NAME}")
        
        st.subheader("🟠 OpenSearch Direct")
        st.write(f"**Host:** {OPENSEARCH_HOST[:20]}...")
        st.write(f"**Index:** {OPENSEARCH_INDEX_NAME}")
        
        # Search settings
        st.subheader("🔍 Search Settings")
        top_k = st.slider("Number of results", min_value=1, max_value=30, value=5)
        
        # Category filter - using top categories from all_categories field
        categories = [
            "All Categories",
            "personal care",
            "snacks", 
            "pantry",
            "beverages",
            "frozen",
            "dairy eggs",
            "household",
            "canned goods",
            "dry goods pasta",
            "produce",
            "bakery",
            "deli",
            "candy chocolate",
            "breakfast",
            "international",
            "ice cream ice",
            "yogurt",
            "alcohol",
            "babies"
        ]
        category_filter = st.selectbox("Filter by category", categories, index=0)
        

        
        # Status indicators
        st.subheader("📊 Status")
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            st.success("✅ Sentence Transformers available")
        else:
            st.error("❌ Sentence Transformers not installed")
            st.code("pip install sentence-transformers")
    
    # Check dependencies
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        st.error("❌ sentence-transformers not installed")
        st.code("pip install sentence-transformers")
        st.stop()
    
    # Initialize search engines
    with st.spinner("Initializing search engines..."):
        s3_search, opensearch_search = initialize_search_engines()
        
        # Load models
        s3_loaded = s3_search.load_model()
        os_loaded = opensearch_search.load_model()
    
    if not s3_loaded or not os_loaded:
        st.error("❌ Failed to initialize search engines")
        st.stop()
    
    st.success("✅ Both search engines initialized successfully")
    
    # Main search interface
    st.header("🔍 Comparative Search")
    
    # Search input
    query = st.text_input(
        "Enter your search query:",
        value=st.session_state.get('search_query', ''),
        placeholder="e.g., chocolate cookies, organic bananas, whole milk",
        help="Search for products using natural language - results will be compared between both methods"
    )
    
    # Search button
    search_button_clicked = st.button("🔍 Compare Search Methods", type="primary")
    
    # Trigger search if button clicked or if there's a query from session state
    if (search_button_clicked or query) and query and len(query.strip()) > 0:
        if category_filter and category_filter != "All Categories":
            st.write(f"**Comparing search results for:** '{query}' **filtered by category:** '{category_filter}'")
        else:
            st.write(f"**Comparing search results for:** '{query}'")
        
        # Generate embedding once for both searches
        with st.spinner("Generating query embedding..."):
            query_embedding = generate_shared_embedding(s3_search, query)
        
        if not query_embedding:
            st.error("❌ Failed to generate query embedding")
            return
        
        st.success(f"✅ Generated {len(query_embedding)}-dimensional embedding")
        
        # Perform searches in parallel using the same embedding
        col1, col2 = st.columns(2)
        
        with col1:
            with st.spinner("Searching S3 Vectors..."):
                s3_results = s3_search.search_products(query, top_k=top_k, query_embedding=query_embedding, category_filter=category_filter)
        
        with col2:
            with st.spinner("Searching OpenSearch..."):
                opensearch_results = opensearch_search.search_products(query, top_k=top_k, query_embedding=query_embedding, category_filter=category_filter)
        
        # Display results
        if s3_results or opensearch_results:
            display_comparison_results(s3_results, opensearch_results)
            
            # Analysis
            if s3_results and opensearch_results:
                analyze_differences(s3_results, opensearch_results)
        else:
            st.warning("No results found from either search method.")
    

    
    # Sample queries
    st.header("💡 Try These Sample Queries")
    sample_queries = [
        "chocolate cookies",
        "organic fruits", 
        "dairy milk",
        "breakfast cereal",
        "frozen pizza",
        "fresh vegetables"
    ]
    
    cols = st.columns(3)
    for i, sample_query in enumerate(sample_queries):
        with cols[i % 3]:
            if st.button(f"🔍 {sample_query}", key=f"sample_{i}"):
                # Update the query in session state and trigger a rerun
                st.session_state['search_query'] = sample_query
                st.rerun()
    


if __name__ == "__main__":
    main()