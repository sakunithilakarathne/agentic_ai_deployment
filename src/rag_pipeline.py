"""
RAG Pipeline for Strategic Plan Synchronization
Enables Q&A over strategic plans, action plans, and analysis results
"""

import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from openai import OpenAI
from pinecone import Pinecone
import time


@dataclass
class DocumentChunk:
    """Represents a chunk of document text"""
    chunk_id: str
    text: str
    source: str  # 'strategic_plan', 'action_plan', 'analysis_results'
    section_type: str  # 'objective', 'action', 'analysis', 'metadata'
    section_title: str
    metadata: Dict


class RAGPipeline:
    """RAG pipeline for answering questions about strategic alignment"""
    
    def __init__(
        self,
        openai_api_key: str,
        pinecone_api_key: str,
        index_name: str = "strategic-alignment"
    ):
        """
        Initialize RAG pipeline
        
        Args:
            openai_api_key: OpenAI API key
            pinecone_api_key: Pinecone API key
            index_name: Name for Pinecone index
        """
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.pinecone_client = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name
        self.embedding_dimension = 3072  # text-embedding-3-large
        
        # Setup index
        self._setup_index()
    
    def _setup_index(self):
        """Create or connect to Pinecone index for RAG"""
        existing_indexes = [idx.name for idx in self.pinecone_client.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating new Pinecone RAG index: {self.index_name}")
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=self.embedding_dimension,
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
            )
            time.sleep(10)
        else:
            print(f"Connecting to existing RAG index: {self.index_name}")
        
        self.index = self.pinecone_client.Index(self.index_name)
    
    def chunk_document(
        self,
        document: Dict,
        doc_type: str,
        chunk_size: int = 500
    ) -> List[DocumentChunk]:
        """
        Split document into chunks for RAG
        
        Args:
            document: Document dict from document_processor
            doc_type: 'strategic_plan' or 'action_plan'
            chunk_size: Target characters per chunk
            
        Returns:
            List of DocumentChunk objects
        """
        print(f"\nChunking {doc_type}...")
        chunks = []
        
        # Add document metadata as a chunk
        metadata_text = f"""
Document: {document.get('title', 'Unknown')}
Organization: {document.get('organization', 'Unknown')}
Planning Period: {document.get('planning_period', 'Unknown')}
Total Budget: ${document.get('total_budget', 0):,.0f}
Total Sections: {len(document.get('sections', []))}
"""
        
        chunks.append(DocumentChunk(
            chunk_id=f"{doc_type}_metadata",
            text=metadata_text.strip(),
            source=doc_type,
            section_type='metadata',
            section_title='Document Overview',
            metadata={
                'document_type': doc_type,
                'title': document.get('title', ''),
                'organization': document.get('organization', '')
            }
        ))
        
        # Process each section
        for idx, section in enumerate(document.get('sections', [])):
            section_id = section.get('id', f'section_{idx}')
            section_title = section.get('title', 'Untitled')
            section_type = section.get('type', 'unknown')
            content = section.get('content', '')
            
            # Create main section chunk
            section_summary = f"""
Title: {section_title}
Type: {section_type.replace('_', ' ').title()}
"""
            
            # Add KPIs if present
            if section.get('kpis'):
                section_summary += f"\nKPIs ({len(section['kpis'])}):\n"
                for kpi in section['kpis'][:5]:  # Top 5
                    kpi_text = kpi.get('metric', 'Unknown')
                    if kpi.get('target'):
                        kpi_text += f": {kpi['target']}{kpi.get('unit', '')}"
                    if kpi.get('deadline'):
                        kpi_text += f" by {kpi['deadline']}"
                    section_summary += f"- {kpi_text}\n"
            
            # Add budget if present
            if section.get('budget'):
                section_summary += f"\nBudget: ${section['budget']:,.0f}"
            
            # Add timeline if present
            if section.get('timeline'):
                section_summary += f"\nTimeline: {section['timeline']}"
            
            # Add initiatives if present
            if section.get('initiatives'):
                section_summary += f"\nInitiatives ({len(section['initiatives'])}):\n"
                for init in section['initiatives'][:3]:
                    section_summary += f"- {init}\n"
            
            # Combine summary with content
            full_text = section_summary + "\n\nDetails:\n" + content
            
            # Split into chunks if too long
            if len(full_text) <= chunk_size * 2:
                # Single chunk
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc_type}_{section_id}",
                    text=full_text,
                    source=doc_type,
                    section_type=section_type,
                    section_title=section_title,
                    metadata={
                        'document_type': doc_type,
                        'section_id': section_id,
                        'budget': section.get('budget', 0),
                        'timeline': section.get('timeline', ''),
                        'kpi_count': len(section.get('kpis', []))
                    }
                ))
            else:
                # Multiple chunks
                # Keep summary in first chunk
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc_type}_{section_id}_summary",
                    text=section_summary,
                    source=doc_type,
                    section_type=section_type,
                    section_title=f"{section_title} (Summary)",
                    metadata={
                        'document_type': doc_type,
                        'section_id': section_id,
                        'is_summary': True
                    }
                ))
                
                # Split content into chunks
                words = content.split()
                current_chunk = []
                current_length = 0
                chunk_num = 0
                
                for word in words:
                    current_chunk.append(word)
                    current_length += len(word) + 1
                    
                    if current_length >= chunk_size:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append(DocumentChunk(
                            chunk_id=f"{doc_type}_{section_id}_chunk_{chunk_num}",
                            text=f"[{section_title} - Part {chunk_num + 1}]\n\n{chunk_text}",
                            source=doc_type,
                            section_type=section_type,
                            section_title=f"{section_title} (Part {chunk_num + 1})",
                            metadata={
                                'document_type': doc_type,
                                'section_id': section_id,
                                'chunk_number': chunk_num
                            }
                        ))
                        current_chunk = []
                        current_length = 0
                        chunk_num += 1
                
                # Add remaining content
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(DocumentChunk(
                        chunk_id=f"{doc_type}_{section_id}_chunk_{chunk_num}",
                        text=f"[{section_title} - Part {chunk_num + 1}]\n\n{chunk_text}",
                        source=doc_type,
                        section_type=section_type,
                        section_title=f"{section_title} (Part {chunk_num + 1})",
                        metadata={
                            'document_type': doc_type,
                            'section_id': section_id,
                            'chunk_number': chunk_num
                        }
                    ))
        
        print(f"  Created {len(chunks)} chunks from {len(document.get('sections', []))} sections")
        return chunks
    
    def chunk_analysis_results(
        self,
        analysis_results: Dict
    ) -> List[DocumentChunk]:
        """
        Convert analysis results into chunks
        
        Args:
            analysis_results: Final synchronization results
            
        Returns:
            List of DocumentChunk objects
        """
        print("\nChunking analysis results...")
        chunks = []
        
        # Overall summary chunk
        summary_text = f"""
SYNCHRONIZATION ANALYSIS SUMMARY

Overall Score: {analysis_results['overall_score']:.1f}/100
Embedding Score: {analysis_results['embedding_score']:.1f}/100
Entity Match Score: {analysis_results['entity_score']:.1f}/100

Objectives Analysis:
- Total Objectives: {analysis_results['summary']['total_objectives']}
- Strong Support: {analysis_results['summary']['objectives_with_strong_support']}
- Weak Support: {analysis_results['summary']['objectives_with_weak_support']}

Entity Coverage:
- Total Strategic Entities: {analysis_results['summary']['total_strategic_entities']}
- Matched: {analysis_results['summary']['matched_entities']}
- Unmatched: {analysis_results['summary']['unmatched_entities']}
"""
        
        chunks.append(DocumentChunk(
            chunk_id="analysis_summary",
            text=summary_text,
            source='analysis_results',
            section_type='analysis',
            section_title='Overall Analysis Summary',
            metadata={
                'document_type': 'analysis',
                'overall_score': analysis_results['overall_score']
            }
        ))
        
        # Strengths chunk
        if analysis_results.get('strengths'):
            strengths_text = "IDENTIFIED STRENGTHS:\n\n"
            for i, strength in enumerate(analysis_results['strengths'], 1):
                strengths_text += f"{i}. {strength}\n\n"
            
            chunks.append(DocumentChunk(
                chunk_id="analysis_strengths",
                text=strengths_text,
                source='analysis_results',
                section_type='analysis',
                section_title='Alignment Strengths',
                metadata={'document_type': 'analysis', 'category': 'strengths'}
            ))
        
        # Weaknesses chunk
        if analysis_results.get('weaknesses'):
            weaknesses_text = "IDENTIFIED WEAKNESSES:\n\n"
            for i, weakness in enumerate(analysis_results['weaknesses'], 1):
                weaknesses_text += f"{i}. {weakness}\n\n"
            
            chunks.append(DocumentChunk(
                chunk_id="analysis_weaknesses",
                text=weaknesses_text,
                source='analysis_results',
                section_type='analysis',
                section_title='Alignment Weaknesses',
                metadata={'document_type': 'analysis', 'category': 'weaknesses'}
            ))
        
        # Recommendations chunk
        if analysis_results.get('recommendations'):
            for i, rec in enumerate(analysis_results['recommendations']):
                rec_text = f"""
RECOMMENDATION {i + 1}
Priority: {rec.get('priority', 'medium').upper()}
Objective: {rec.get('objective', 'General')}
Current Score: {rec.get('current_score', 'N/A')}

Recommended Actions:
"""
                for action in rec.get('actions', []):
                    rec_text += f"- {action}\n"
                
                if rec.get('expected_impact'):
                    rec_text += f"\nExpected Impact:\n{rec['expected_impact']}"
                
                chunks.append(DocumentChunk(
                    chunk_id=f"analysis_recommendation_{i}",
                    text=rec_text,
                    source='analysis_results',
                    section_type='analysis',
                    section_title=f"Recommendation: {rec.get('objective', 'General')}",
                    metadata={
                        'document_type': 'analysis',
                        'category': 'recommendation',
                        'priority': rec.get('priority', 'medium')
                    }
                ))
        
        # Per-objective analysis chunks
        for obj in analysis_results.get('objective_synchronizations', []):
            obj_text = f"""
OBJECTIVE ANALYSIS: {obj['objective_title']}

Scores:
- Combined Score: {obj['combined_score']:.1f}/100
- Embedding Score: {obj['embedding_score']:.1f}/100
- Entity Matches: {obj['entity_matches']}
- Status: {'Strong Support' if obj['has_strong_support'] else 'Weak Support'}

Top Matching Actions:
"""
            for action in obj.get('top_matching_actions', [])[:3]:
                obj_text += f"{action['rank']}. {action['action_title']} (similarity: {action['similarity_score']:.2f})\n"
            
            if obj.get('gaps'):
                obj_text += "\nIdentified Gaps:\n"
                for gap in obj['gaps']:
                    obj_text += f"- {gap}\n"
            
            chunks.append(DocumentChunk(
                chunk_id=f"analysis_objective_{obj['objective_id']}",
                text=obj_text,
                source='analysis_results',
                section_type='analysis',
                section_title=f"Analysis: {obj['objective_title']}",
                metadata={
                    'document_type': 'analysis',
                    'category': 'objective_analysis',
                    'objective_id': obj['objective_id'],
                    'combined_score': obj['combined_score']
                }
            ))
        
        print(f"  Created {len(chunks)} analysis chunks")
        return chunks
    
    def embed_chunk(self, text: str) -> List[float]:
        """Generate embedding for a text chunk"""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=text[:8000]  # Limit to 8K chars to avoid token limit
        )
        return response.data[0].embedding
    
    def build_vector_store(
        self,
        strategic_doc: Dict,
        action_doc: Dict,
        analysis_results: Dict
    ):
        """
        Build complete vector store with all documents
        
        Args:
            strategic_doc: Strategic plan document
            action_doc: Action plan document
            analysis_results: Final synchronization results
        """
        print("\n" + "="*70)
        print("BUILDING RAG VECTOR STORE")
        print("="*70)
        
        # Chunk all documents
        strategic_chunks = self.chunk_document(strategic_doc, 'strategic_plan')
        action_chunks = self.chunk_document(action_doc, 'action_plan')
        analysis_chunks = self.chunk_analysis_results(analysis_results)
        
        all_chunks = strategic_chunks + action_chunks + analysis_chunks
        
        print(f"\nTotal chunks to index: {len(all_chunks)}")
        print(f"  Strategic Plan: {len(strategic_chunks)}")
        print(f"  Action Plan: {len(action_chunks)}")
        print(f"  Analysis Results: {len(analysis_chunks)}")
        
        # Embed and index chunks
        print("\nEmbedding and indexing chunks...")
        vectors = []
        
        for i, chunk in enumerate(all_chunks):
            if (i + 1) % 10 == 0:
                print(f"  Processing chunk {i + 1}/{len(all_chunks)}...")
            
            # Generate embedding
            embedding = self.embed_chunk(chunk.text)
            
            # Prepare vector
            vector = {
                'id': chunk.chunk_id,
                'values': embedding,
                'metadata': {
                    'text': chunk.text[:1000],  # Store first 1000 chars in metadata
                    'source': chunk.source,
                    'section_type': chunk.section_type,
                    'section_title': chunk.section_title,
                    **chunk.metadata
                }
            }
            
            vectors.append(vector)
            
            # Batch upsert every 50 chunks
            if len(vectors) >= 50:
                self.index.upsert(vectors=vectors, namespace="rag")
                vectors = []
        
        # Upsert remaining vectors
        if vectors:
            self.index.upsert(vectors=vectors, namespace="rag")
        
        print(f"\n✓ Indexed {len(all_chunks)} chunks in vector store")
    
    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        filter_source: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            filter_source: Optional filter ('strategic_plan', 'action_plan', 'analysis_results')
            
        Returns:
            List of relevant chunks with metadata
        """
        # Embed query
        query_embedding = self.embed_chunk(query)
        
        # Build filter
        filter_dict = {}
        if filter_source:
            filter_dict['source'] = filter_source
        
        # Query vector store
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace="rag",
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        
        # Extract relevant chunks
        contexts = []
        for match in results.matches:
            contexts.append({
                'text': match.metadata.get('text', ''),
                'source': match.metadata.get('source', ''),
                'section_title': match.metadata.get('section_title', ''),
                'similarity_score': match.score
            })
        
        return contexts
    
    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        include_sources: bool = True
    ) -> Dict:
        """
        Answer a question using RAG
        
        Args:
            question: User question
            top_k: Number of context chunks to retrieve
            include_sources: Whether to include source citations
            
        Returns:
            Dict with answer and sources
        """
        print(f"\n🔍 Question: {question}")
        print("Retrieving relevant context...")
        
        # Retrieve context
        contexts = self.retrieve_context(question, top_k=top_k)
        
        if not contexts:
            return {
                'answer': "I couldn't find relevant information to answer this question. Please try rephrasing or ask about specific objectives, actions, or analysis results.",
                'sources': []
            }
        
        # Build context string
        context_text = "\n\n---\n\n".join([
            f"[Source: {ctx['source']} - {ctx['section_title']}]\n{ctx['text']}"
            for ctx in contexts
        ])
        
        # Generate answer with LLM
        prompt = f"""You are an expert analyst helping stakeholders understand strategic plan synchronization analysis.

Use the following context to answer the question. Be specific and reference the data provided.

CONTEXT:
{context_text}

QUESTION: {question}

Provide a clear, detailed answer based on the context. If the context doesn't contain enough information, say so and suggest what additional information might be helpful.

Answer:"""
        
        print("Generating answer with GPT-4...")
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a strategic planning expert helping analyze plan synchronization."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        
        # Prepare sources
        sources = []
        if include_sources:
            for ctx in contexts:
                sources.append({
                    'source': ctx['source'],
                    'section': ctx['section_title'],
                    'similarity': f"{ctx['similarity_score']:.2f}"
                })
        
        return {
            'answer': answer,
            'sources': sources,
            'num_contexts_used': len(contexts)
        }
    
    def clear_vector_store(self):
        """Clear all vectors from RAG index"""
        print("\nClearing RAG vector store...")
        self.index.delete(delete_all=True, namespace="rag")
        print("✓ Vector store cleared")


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize RAG pipeline
    rag = RAGPipeline(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        pinecone_api_key=os.getenv('PINECONE_API_KEY'),
        index_name="strategic-rag"
    )
    
    # Load documents
    with open('strategic_plan.json', 'r') as f:
        strategic_doc = json.load(f)
    
    with open('action_plan.json', 'r') as f:
        action_doc = json.load(f)
    
    with open('final_synchronization_results.json', 'r') as f:
        analysis_results = json.load(f)
    
    # Build vector store
    rag.build_vector_store(
        strategic_doc=strategic_doc,
        action_doc=action_doc,
        analysis_results=analysis_results
    )
    
    # Test with sample questions
    questions = [
        "Why is the Digital Transformation objective scoring high?",
        "What are the main weaknesses in the Risk Management objective?",
        "What specific actions are recommended for improving alignment?"
    ]
    
    print("\n" + "="*70)
    print("TESTING RAG PIPELINE")
    print("="*70)
    
    for question in questions:
        result = rag.answer_question(question)
        print(f"\n{'='*70}")
        print(f"Q: {question}")
        print(f"{'='*70}")
        print(f"A: {result['answer']}")
        print(f"\nSources used: {result['num_contexts_used']}")
    
    print("\n✓ RAG pipeline test complete!")
