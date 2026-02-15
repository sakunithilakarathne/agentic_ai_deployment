"""
Embedding Analyzer for Strategic Plan Synchronization
Uses OpenAI embeddings and Pinecone vector database for semantic similarity
"""

import os
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import json
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec


@dataclass
class SimilarityMatch:
    """Represents a similarity match between strategic objective and action"""
    strategic_id: str
    strategic_title: str
    action_id: str
    action_title: str
    similarity_score: float
    rank: int  # 1 = best match, 2 = second best, etc.


@dataclass
class ObjectiveAlignment:
    """Alignment analysis for a single strategic objective"""
    objective_id: str
    objective_title: str
    best_match_score: float
    top_matches: List[SimilarityMatch]
    has_support: bool  # True if similarity > threshold
    

@dataclass
class EmbeddingAnalysisResult:
    """Complete embedding analysis results"""
    overall_score: float  # 0-100
    objective_alignments: List[ObjectiveAlignment]
    average_similarity: float
    objectives_with_support: int
    objectives_without_support: int
    threshold: float


class EmbeddingAnalyzer:
    """Analyzes semantic similarity using embeddings"""
    
    def __init__(
        self, 
        openai_api_key: str,
        pinecone_api_key: str,
        index_name: str = "strategic-alignment",
        similarity_threshold: float = 0.70
    ):
        """
        Initialize the analyzer
        
        Args:
            openai_api_key: OpenAI API key
            pinecone_api_key: Pinecone API key
            index_name: Name for Pinecone index
            similarity_threshold: Minimum score to consider aligned (0-1)
        """
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.pinecone_client = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name
        self.similarity_threshold = similarity_threshold
        self.embedding_dimension = 3072  # text-embedding-3-large
        
        # Initialize or connect to index
        self._setup_index()
        
    def _setup_index(self):
        """Create or connect to Pinecone index"""
        existing_indexes = [idx.name for idx in self.pinecone_client.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating new Pinecone index: {self.index_name}")
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=self.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            # Wait for index to be ready
            time.sleep(10)
        else:
            print(f"Connecting to existing index: {self.index_name}")
        
        self.index = self.pinecone_client.Index(self.index_name)
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenAI
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    def index_strategic_plan(self, strategic_doc: Dict):
        """
        Index strategic plan objectives in Pinecone
        
        Args:
            strategic_doc: Document dict from document_processor
        """
        print("\nIndexing Strategic Plan objectives...")
        vectors = []
        
        for section in strategic_doc['sections']:
            if section['type'] == 'strategic_objective':
                # Combine title and content for better embedding
                text_to_embed = f"{section['title']}. {section['content'][:1000]}"
                
                print(f"  Embedding: {section['title'][:50]}...")
                embedding = self.generate_embedding(text_to_embed)
                
                # Prepare metadata
                metadata = {
                    'id': section['id'],
                    'title': section['title'],
                    'type': 'strategic_objective',
                    'document': 'strategic_plan',
                    'budget': float(section.get('budget') or 0),
                    'timeline': section.get('timeline') or '',
                    'kpi_count': len(section.get('kpis', []))
                }
                
                vectors.append({
                    'id': f"sp_{section['id']}",
                    'values': embedding,
                    'metadata': metadata
                })
        
        # Upsert to Pinecone
        if vectors:
            self.index.upsert(
                vectors=vectors,
                namespace="strategic_plan"
            )
            print(f"✓ Indexed {len(vectors)} strategic objectives")
        else:
            print("⚠ No strategic objectives found to index")
    
    def index_action_plan(self, action_doc: Dict):
        """
        Index action plan items in Pinecone
        
        Args:
            action_doc: Document dict from document_processor
        """
        print("\nIndexing Action Plan items...")
        vectors = []
        
        for section in action_doc['sections']:
            if section['type'] == 'action_item':
                # Combine title and content for better embedding
                text_to_embed = f"{section['title']}. {section['content'][:1000]}"
                
                print(f"  Embedding: {section['title'][:50]}...")
                embedding = self.generate_embedding(text_to_embed)
                
                # Prepare metadata
                metadata = {
                    'id': section['id'],
                    'title': section['title'],
                    'type': 'action_item',
                    'document': 'action_plan',
                    'budget': float(section.get('budget') or 0),
                    'timeline': section.get('timeline') or '',
                    'priority': section.get('priority') or '',
                    'kpi_count': len(section.get('kpis', []))
                }
                
                vectors.append({
                    'id': f"ap_{section['id']}",
                    'values': embedding,
                    'metadata': metadata
                })
        
        # Upsert to Pinecone
        if vectors:
            self.index.upsert(
                vectors=vectors,
                namespace="action_plan"
            )
            print(f"✓ Indexed {len(vectors)} action items")
        else:
            print("⚠ No action items found to index")
    
    def find_similar_actions(
        self, 
        objective_embedding: List[float],
        objective_id: str,
        objective_title: str,
        top_k: int = 5
    ) -> List[SimilarityMatch]:
        """
        Find most similar action items for a strategic objective
        
        Args:
            objective_embedding: Embedding vector for the objective
            objective_id: ID of the objective
            objective_title: Title of the objective
            top_k: Number of top matches to return
            
        Returns:
            List of SimilarityMatch objects
        """
        # Query Pinecone for similar actions
        results = self.index.query(
            vector=objective_embedding,
            top_k=top_k,
            namespace="action_plan",
            include_metadata=True
        )
        
        # Convert to SimilarityMatch objects
        matches = []
        for rank, match in enumerate(results.matches, 1):
            similarity_match = SimilarityMatch(
                strategic_id=objective_id,
                strategic_title=objective_title,
                action_id=match.metadata['id'],
                action_title=match.metadata['title'],
                similarity_score=match.score,
                rank=rank
            )
            matches.append(similarity_match)
        
        return matches
    
    def analyze_synchronization(
        self,
        strategic_doc: Dict,
        action_doc: Dict,
        top_k: int = 5
    ) -> EmbeddingAnalysisResult:
        """
        Perform complete synchronization analysis
        
        Args:
            strategic_doc: Strategic plan document dict
            action_doc: Action plan document dict
            top_k: Number of top matches to find per objective
            
        Returns:
            EmbeddingAnalysisResult with complete analysis
        """
        print("\n" + "="*60)
        print("EMBEDDING ANALYSIS - SYNCHRONIZATION ASSESSMENT")
        print("="*60)
        
        # Index both documents
        self.index_strategic_plan(strategic_doc)
        self.index_action_plan(action_doc)
        
        print("\n" + "-"*60)
        print("Analyzing objective-action alignment...")
        print("-"*60)
        
        objective_alignments = []
        all_best_scores = []
        objectives_with_support = 0
        objectives_without_support = 0
        
        # Analyze each strategic objective
        strategic_objectives = [
            s for s in strategic_doc['sections'] 
            if s['type'] == 'strategic_objective'
        ]
        
        for i, objective in enumerate(strategic_objectives, 1):
            print(f"\n[{i}/{len(strategic_objectives)}] Analyzing: {objective['title']}")
            
            # Generate embedding for objective
            text_to_embed = f"{objective['title']}. {objective['content'][:1000]}"
            objective_embedding = self.generate_embedding(text_to_embed)
            
            # Find similar actions
            matches = self.find_similar_actions(
                objective_embedding=objective_embedding,
                objective_id=objective['id'],
                objective_title=objective['title'],
                top_k=top_k
            )
            
            # Determine if objective has supporting actions
            best_score = matches[0].similarity_score if matches else 0.0
            has_support = best_score >= self.similarity_threshold
            
            if has_support:
                objectives_with_support += 1
                print(f"  ✓ Supported (best match: {best_score:.3f})")
            else:
                objectives_without_support += 1
                print(f"  ⚠ Weak support (best match: {best_score:.3f})")
            
            # Show top 3 matches
            for match in matches[:3]:
                print(f"    {match.rank}. {match.action_title[:60]}... ({match.similarity_score:.3f})")
            
            # Store alignment info
            alignment = ObjectiveAlignment(
                objective_id=objective['id'],
                objective_title=objective['title'],
                best_match_score=best_score,
                top_matches=matches,
                has_support=has_support
            )
            objective_alignments.append(alignment)
            all_best_scores.append(best_score)
        
        # Calculate overall metrics
        average_similarity = sum(all_best_scores) / len(all_best_scores) if all_best_scores else 0.0
        overall_score = average_similarity * 100  # Convert to 0-100 scale
        
        result = EmbeddingAnalysisResult(
            overall_score=overall_score,
            objective_alignments=objective_alignments,
            average_similarity=average_similarity,
            objectives_with_support=objectives_with_support,
            objectives_without_support=objectives_without_support,
            threshold=self.similarity_threshold
        )
        
        # Print summary
        self._print_summary(result)
        
        return result
    
    def _print_summary(self, result: EmbeddingAnalysisResult):
        """Print analysis summary"""
        print("\n" + "="*60)
        print("EMBEDDING ANALYSIS RESULTS")
        print("="*60)
        print(f"\nOverall Synchronization Score: {result.overall_score:.1f}/100")
        print(f"Average Similarity: {result.average_similarity:.3f}")
        print(f"Similarity Threshold: {result.threshold:.2f}")
        print(f"\nObjectives with Strong Support: {result.objectives_with_support}")
        print(f"Objectives with Weak Support: {result.objectives_without_support}")
        
        # Show objectives without support
        if result.objectives_without_support > 0:
            print("\n⚠ Objectives needing attention:")
            for alignment in result.objective_alignments:
                if not alignment.has_support:
                    print(f"  - {alignment.objective_title}")
                    print(f"    Best match score: {alignment.best_match_score:.3f}")
    
    def save_results(self, result: EmbeddingAnalysisResult, output_path: str):
        """Save analysis results to JSON file"""
        # Convert to dict
        result_dict = {
            'overall_score': result.overall_score,
            'average_similarity': result.average_similarity,
            'objectives_with_support': result.objectives_with_support,
            'objectives_without_support': result.objectives_without_support,
            'threshold': result.threshold,
            'objective_alignments': [
                {
                    'objective_id': a.objective_id,
                    'objective_title': a.objective_title,
                    'best_match_score': a.best_match_score,
                    'has_support': a.has_support,
                    'top_matches': [
                        {
                            'action_id': m.action_id,
                            'action_title': m.action_title,
                            'similarity_score': m.similarity_score,
                            'rank': m.rank
                        } for m in a.top_matches
                    ]
                } for a in result.objective_alignments
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"\n✓ Results saved to {output_path}")
    
    def clear_index(self):
        """Clear all vectors from the index"""
        print("\nClearing Pinecone index...")
        self.index.delete(delete_all=True, namespace="strategic_plan")
        self.index.delete(delete_all=True, namespace="action_plan")
        print("✓ Index cleared")


# Example usage
if __name__ == "__main__":
    import json
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Initialize analyzer
    analyzer = EmbeddingAnalyzer(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        pinecone_api_key=os.getenv('PINECONE_API_KEY'),
        index_name="strategic-alignment",
        similarity_threshold=0.70
    )
    
    # Load processed documents
    with open('strategic_plan.json', 'r') as f:
        strategic_doc = json.load(f)
    
    with open('action_plan.json', 'r') as f:
        action_doc = json.load(f)
    
    # Analyze synchronization
    result = analyzer.analyze_synchronization(
        strategic_doc=strategic_doc,
        action_doc=action_doc,
        top_k=5
    )
    
    # Save results
    analyzer.save_results(result, 'embedding_analysis_results.json')
    
    print("\n✓ Embedding analysis complete!")
