"""
Scoring Engine - Combines Embedding and Entity Analysis with LLM-Powered Insights
Produces final synchronization score and comprehensive, context-aware recommendations
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from openai import OpenAI


@dataclass
class ObjectiveSynchronization:
    """Synchronization details for a single strategic objective"""
    objective_id: str
    objective_title: str
    embedding_score: float
    entity_matches: int
    entity_score: float
    combined_score: float
    top_matching_actions: List[Dict]
    has_strong_support: bool
    gaps: List[str]


@dataclass
class OverallSynchronization:
    """Complete synchronization assessment results"""
    overall_score: float  # 0-100
    embedding_score: float
    entity_score: float
    
    # Summary metrics
    total_objectives: int
    objectives_with_strong_support: int
    objectives_with_weak_support: int
    
    # Detailed breakdowns
    objective_synchronizations: List[ObjectiveSynchronization]
    
    # Entity details
    total_strategic_entities: int
    matched_entities: int
    unmatched_entities: int
    
    # Metadata
    assessment_date: str
    strategic_plan_title: str
    action_plan_title: str
    
    # Recommendations
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[Dict]


class LLMScoringEngine:
    """Combines multiple analysis methods into final score with LLM-powered insights"""
    
    def __init__(
        self,
        openai_api_key: str,
        embedding_weight: float = 0.60,
        entity_weight: float = 0.40,
        strong_support_threshold: float = 75.0
    ):
        """
        Initialize scoring engine
        
        Args:
            openai_api_key: OpenAI API key for LLM insights
            embedding_weight: Weight for embedding analysis (0-1)
            entity_weight: Weight for entity matching (0-1)
            strong_support_threshold: Minimum score for "strong support" (0-100)
        """
        # Validate weights sum to 1.0
        total = embedding_weight + entity_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        
        self.embedding_weight = embedding_weight
        self.entity_weight = entity_weight
        self.strong_support_threshold = strong_support_threshold
        self.openai_client = OpenAI(api_key=openai_api_key)
    
    def combine_scores(
        self,
        embedding_results: Dict,
        entity_results: Dict,
        strategic_doc: Dict,
        action_doc: Dict
    ) -> OverallSynchronization:
        """
        Combine all analysis results into final assessment
        
        Args:
            embedding_results: Results from embedding_analyzer
            entity_results: Results from entity_extractor
            strategic_doc: Strategic plan document
            action_doc: Action plan document
            
        Returns:
            OverallSynchronization with complete assessment
        """
        print("\n" + "="*70)
        print("COMBINING ANALYSIS RESULTS")
        print("="*70)
        
        # Calculate overall score
        overall_score = (
            self.embedding_weight * embedding_results['overall_score'] +
            self.entity_weight * entity_results['overall_score']
        )
        
        print(f"\nWeighted Score Calculation:")
        print(f"  Embedding ({self.embedding_weight*100:.0f}%): {embedding_results['overall_score']:.1f}")
        print(f"  Entity    ({self.entity_weight*100:.0f}%): {entity_results['overall_score']:.1f}")
        print(f"  ──────────────────────────────")
        print(f"  Overall Score: {overall_score:.1f}/100")
        
        # Analyze each objective
        objective_syncs = self._analyze_objectives(
            embedding_results,
            entity_results
        )
        
        # Count support levels
        strong_support = sum(1 for obj in objective_syncs if obj.has_strong_support)
        weak_support = len(objective_syncs) - strong_support
        
        print("\n" + "="*70)
        print("GENERATING LLM-POWERED INSIGHTS")
        print("="*70)
        
        # Generate LLM-powered insights
        print("\n🤖 Analyzing strengths with GPT-4...")
        strengths = self._identify_strengths_llm(
            objective_syncs,
            embedding_results,
            entity_results,
            overall_score
        )
        
        print("🤖 Analyzing weaknesses with GPT-4...")
        weaknesses = self._identify_weaknesses_llm(
            objective_syncs,
            entity_results,
            overall_score
        )
        
        print("🤖 Generating recommendations with GPT-4...")
        recommendations = self._generate_recommendations_llm(
            objective_syncs,
            weaknesses,
            entity_results,
            overall_score
        )
        
        # Create final result
        result = OverallSynchronization(
            overall_score=overall_score,
            embedding_score=embedding_results['overall_score'],
            entity_score=entity_results['overall_score'],
            total_objectives=len(objective_syncs),
            objectives_with_strong_support=strong_support,
            objectives_with_weak_support=weak_support,
            objective_synchronizations=objective_syncs,
            total_strategic_entities=entity_results['total_strategic_entities'],
            matched_entities=entity_results['matched_entities'],
            unmatched_entities=entity_results['unmatched_entities'],
            assessment_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            strategic_plan_title=strategic_doc.get('title', 'Strategic Plan'),
            action_plan_title=action_doc.get('title', 'Action Plan'),
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations
        )
        
        self._print_summary(result)
        
        return result
    
    def _analyze_objectives(
        self,
        embedding_results: Dict,
        entity_results: Dict
    ) -> List[ObjectiveSynchronization]:
        """Analyze synchronization for each objective"""
        
        print("\nAnalyzing per-objective synchronization...")
        
        objective_syncs = []
        
        # Map entity matches by objective
        entity_matches_by_obj = {}
        for match in entity_results.get('entity_matches', []):
            obj_title = match['strategic_entity']['source']
            if obj_title not in entity_matches_by_obj:
                entity_matches_by_obj[obj_title] = []
            entity_matches_by_obj[obj_title].append(match)
        
        # Process each objective
        for obj_alignment in embedding_results.get('objective_alignments', []):
            obj_title = obj_alignment['objective_title']
            
            # Get embedding score
            embedding_score = obj_alignment['best_match_score'] * 100
            
            # Get entity matches for this objective
            obj_entity_matches = entity_matches_by_obj.get(obj_title, [])
            entity_match_count = len(obj_entity_matches)
            
            # Calculate entity score for this objective
            entity_score = min(entity_match_count * 20, 100)
            
            # Combined score for this objective
            combined_score = (
                self.embedding_weight * embedding_score +
                self.entity_weight * entity_score
            )
            
            # Determine support level
            has_strong_support = combined_score >= self.strong_support_threshold
            
            # Identify gaps
            gaps = []
            if embedding_score < 70:
                gaps.append("Low semantic similarity - action may not address objective intent")
            if entity_match_count == 0:
                gaps.append("No explicit KPIs/targets matched in action plan")
            if not obj_alignment['has_support']:
                gaps.append(f"Best match score ({obj_alignment['best_match_score']:.2f}) below threshold")
            
            # Get top matching actions
            top_actions = [
                {
                    'action_id': m['action_id'],
                    'action_title': m['action_title'],
                    'similarity_score': m['similarity_score'],
                    'rank': m['rank']
                }
                for m in obj_alignment.get('top_matches', [])[:3]
            ]
            
            obj_sync = ObjectiveSynchronization(
                objective_id=obj_alignment['objective_id'],
                objective_title=obj_title,
                embedding_score=embedding_score,
                entity_matches=entity_match_count,
                entity_score=entity_score,
                combined_score=combined_score,
                top_matching_actions=top_actions,
                has_strong_support=has_strong_support,
                gaps=gaps
            )
            
            objective_syncs.append(obj_sync)
            
            status = "✓" if has_strong_support else "⚠"
            print(f"  {status} {obj_title[:50]}: {combined_score:.1f}")
        
        return objective_syncs
    
    def _identify_strengths_llm(
        self,
        objective_syncs: List[ObjectiveSynchronization],
        embedding_results: Dict,
        entity_results: Dict,
        overall_score: float
    ) -> List[str]:
        """Use LLM to identify specific strengths with evidence"""
        
        # Prepare data summary for LLM
        strong_objectives = [obj for obj in objective_syncs if obj.has_strong_support]
        
        data_summary = f"""
ANALYSIS RESULTS:
- Overall Score: {overall_score:.1f}/100
- Embedding Score: {embedding_results['overall_score']:.1f}/100
- Entity Match Score: {entity_results['overall_score']:.1f}/100
- Total Objectives: {len(objective_syncs)}
- Strong Support: {len(strong_objectives)}/{len(objective_syncs)}
- Entity Match Rate: {entity_results['match_rate']:.1f}%

TOP PERFORMING OBJECTIVES:
"""
        
        # Add top 3 strongest objectives
        top_objectives = sorted(objective_syncs, key=lambda x: x.combined_score, reverse=True)[:3]
        for obj in top_objectives:
            data_summary += f"\n- '{obj.objective_title}': {obj.combined_score:.1f}/100"
            data_summary += f"\n  Embedding: {obj.embedding_score:.1f}, Entity Matches: {obj.entity_matches}"
            if obj.top_matching_actions:
                data_summary += f"\n  Best Action: {obj.top_matching_actions[0]['action_title']}"
        
        prompt = f"""Analyze this strategic plan synchronization data and return ONLY a JSON object.

{data_summary}

Return format (NO other text, ONLY this JSON):
{{
  "strengths": [
    "string 1",
    "string 2",
    "string 3"
  ]
}}

Requirements for each strength:
1. Reference concrete data (scores, numbers, specific objectives)
2. Explain WHY it's a strength
3. Be specific and evidence-based

Generate 3-5 strengths based on the data above.
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a strategic planning expert analyzing plan synchronization."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            strengths = result.get('strengths', [])
            
            if not strengths:
                # Fallback if JSON parsing fails
                strengths = [f"{len(strong_objectives)}/{len(objective_syncs)} strategic objectives demonstrate strong alignment"]
            
            return strengths
            
        except Exception as e:
            print(f"  ⚠ LLM call failed: {e}")
            # Fallback to basic summary
            return [f"{len(strong_objectives)}/{len(objective_syncs)} strategic objectives have strong supporting actions"]
    
    def _identify_weaknesses_llm(
        self,
        objective_syncs: List[ObjectiveSynchronization],
        entity_results: Dict,
        overall_score: float
    ) -> List[str]:
        """Use LLM to identify specific weaknesses with evidence"""
        
        weak_objectives = [obj for obj in objective_syncs if not obj.has_strong_support]
        
        data_summary = f"""
IDENTIFIED ISSUES:
- Overall Score: {overall_score:.1f}/100
- Weak Objectives: {len(weak_objectives)}/{len(objective_syncs)}
- Unmatched Strategic Entities: {entity_results['unmatched_entities']}/{entity_results['total_strategic_entities']}

WEAKEST OBJECTIVES:
"""
        
        # Add weakest objectives
        weakest = sorted(objective_syncs, key=lambda x: x.combined_score)[:3]
        for obj in weakest:
            data_summary += f"\n- '{obj.objective_title}': {obj.combined_score:.1f}/100"
            data_summary += f"\n  Embedding: {obj.embedding_score:.1f}, Entity Matches: {obj.entity_matches}"
            if obj.gaps:
                data_summary += f"\n  Gaps: {', '.join(obj.gaps[:2])}"
        
        # Add unmatched entities summary
        if entity_results.get('unmatched_strategic_entities'):
            unmatched_by_type = {}
            for entity in entity_results['unmatched_strategic_entities'][:10]:
                entity_type = entity['type']
                unmatched_by_type[entity_type] = unmatched_by_type.get(entity_type, 0) + 1
            
            data_summary += "\n\nUNMATCHED ENTITIES BY TYPE:"
            for entity_type, count in unmatched_by_type.items():
                data_summary += f"\n- {entity_type}: {count}"
        
        prompt = f"""Analyze this strategic plan synchronization data and return ONLY a JSON object.

{data_summary}

Return format (NO other text, ONLY this JSON):
{{
  "weaknesses": [
    "string 1",
    "string 2"
  ]
}}

Requirements:
1. Reference concrete data
2. Explain the IMPACT
3. Only identify genuine weaknesses (if score >80, focus on minor improvements)

If no significant weaknesses, return empty array: {{"weaknesses": []}}

Generate 0-5 weaknesses based on the data above.
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a strategic planning expert analyzing plan synchronization."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            weaknesses = result.get('weaknesses', [])
            
            # If no weaknesses and score is high, return empty
            if not weaknesses and overall_score > 85:
                return []
            
            return weaknesses
            
        except Exception as e:
            print(f"  ⚠ LLM call failed: {e}")
            # Fallback
            if weak_objectives:
                return [f"{len(weak_objectives)} strategic objectives lack strong supporting actions"]
            return []
    
    def _generate_recommendations_llm(
        self,
        objective_syncs: List[ObjectiveSynchronization],
        weaknesses: List[str],
        entity_results: Dict,
        overall_score: float
    ) -> List[Dict]:
        """Use LLM to generate specific, actionable recommendations"""
        
        weak_objectives = [obj for obj in objective_syncs if not obj.has_strong_support]
        
        # If score is very high and no weak objectives, skip recommendations
        if overall_score > 90 and not weak_objectives:
            return []
        
        data_summary = f"""
CURRENT STATE:
- Overall Score: {overall_score:.1f}/100
- Weak Objectives: {len(weak_objectives)}/{len(objective_syncs)}

IDENTIFIED WEAKNESSES:
{chr(10).join(f"- {w}" for w in weaknesses[:5])}

WEAKEST OBJECTIVES (need attention):
"""
        
        # Add details for weak objectives
        for obj in sorted(weak_objectives, key=lambda x: x.combined_score)[:3]:
            data_summary += f"\n\nObjective: '{obj.objective_title}'"
            data_summary += f"\n- Score: {obj.combined_score:.1f}/100"
            data_summary += f"\n- Embedding Score: {obj.embedding_score:.1f}/100"
            data_summary += f"\n- Entity Matches: {obj.entity_matches}"
            if obj.gaps:
                data_summary += f"\n- Gaps: {', '.join(obj.gaps)}"
            if obj.top_matching_actions:
                data_summary += f"\n- Best Matching Action: {obj.top_matching_actions[0]['action_title']} ({obj.top_matching_actions[0]['similarity_score']:.2f})"
        
        # Add sample unmatched entities
        if entity_results.get('unmatched_strategic_entities'):
            data_summary += "\n\nSAMPLE UNMATCHED ENTITIES:"
            for entity in entity_results['unmatched_strategic_entities'][:5]:
                data_summary += f"\n- [{entity['type']}] {entity['text'][:60]}"
        
        prompt = f"""Generate actionable recommendations and return ONLY a JSON object.

{data_summary}

Return format (NO other text, ONLY this JSON):
{{
  "recommendations": [
    {{
      "priority": "high",
      "objective": "objective name",
      "current_score": 62.5,
      "actions": ["action 1", "action 2"],
      "expected_impact": "impact description"
    }}
  ]
}}

Generate 3-5 specific, actionable recommendations. Use actual objective names and scores from the data.

If score is >90 and no weak objectives, return empty array: {{"recommendations": []}}
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a strategic planning consultant specializing in finance organizations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            recommendations = result.get('recommendations', [])
            
            return recommendations
            
        except Exception as e:
            print(f"  ⚠ LLM call failed: {e}")
            # Fallback to basic recommendations
            if weak_objectives:
                return [{
                    'priority': 'high',
                    'objective': weak_objectives[0].objective_title,
                    'current_score': weak_objectives[0].combined_score,
                    'actions': ['Review and strengthen action plan to better address this objective']
                }]
            return []
    
    def _print_summary(self, result: OverallSynchronization):
        """Print comprehensive summary"""
        
        print("\n" + "="*70)
        print("FINAL SYNCHRONIZATION ASSESSMENT")
        print("="*70)
        
        print(f"\n📊 Overall Synchronization Score: {result.overall_score:.1f}/100")
        
        # Interpretation
        if result.overall_score >= 90:
            interpretation = "Excellent - Strong alignment across all objectives"
        elif result.overall_score >= 75:
            interpretation = "Good - Minor gaps that should be addressed"
        elif result.overall_score >= 60:
            interpretation = "Moderate - Significant improvements needed"
        else:
            interpretation = "Poor - Major misalignment requiring urgent attention"
        
        print(f"   {interpretation}")
        
        print(f"\n📈 Component Scores:")
        print(f"   Embedding Analysis: {result.embedding_score:.1f}/100")
        print(f"   Entity Matching:    {result.entity_score:.1f}/100")
        
        print(f"\n🎯 Objective Support:")
        print(f"   Strong Support: {result.objectives_with_strong_support}/{result.total_objectives}")
        print(f"   Weak Support:   {result.objectives_with_weak_support}/{result.total_objectives}")
        
        print(f"\n🏷️ Entity Coverage:")
        print(f"   Matched:   {result.matched_entities}/{result.total_strategic_entities}")
        print(f"   Unmatched: {result.unmatched_entities}/{result.total_strategic_entities}")
        
        print(f"\n✅ Strengths ({len(result.strengths)}):")
        for strength in result.strengths[:3]:
            print(f"   • {strength[:100]}{'...' if len(strength) > 100 else ''}")
        
        if result.weaknesses:
            print(f"\n⚠️  Weaknesses ({len(result.weaknesses)}):")
            for weakness in result.weaknesses[:3]:
                print(f"   • {weakness[:100]}{'...' if len(weakness) > 100 else ''}")
        
        if result.recommendations:
            print(f"\n💡 Top Recommendations ({len(result.recommendations)}):")
            for i, rec in enumerate(result.recommendations[:2], 1):
                print(f"\n   {i}. [{rec['priority'].upper()}] {rec.get('objective', 'General')}")
                if 'current_score' in rec:
                    print(f"      Current Score: {rec['current_score']:.1f}/100")
                for action in rec.get('actions', [])[:2]:
                    print(f"      → {action[:90]}{'...' if len(action) > 90 else ''}")
    
    def save_results(self, result: OverallSynchronization, output_path: str):
        """Save complete results to JSON"""
        
        # Convert to dict
        result_dict = {
            'overall_score': result.overall_score,
            'embedding_score': result.embedding_score,
            'entity_score': result.entity_score,
            'assessment_date': result.assessment_date,
            'strategic_plan': result.strategic_plan_title,
            'action_plan': result.action_plan_title,
            'summary': {
                'total_objectives': result.total_objectives,
                'objectives_with_strong_support': result.objectives_with_strong_support,
                'objectives_with_weak_support': result.objectives_with_weak_support,
                'total_strategic_entities': result.total_strategic_entities,
                'matched_entities': result.matched_entities,
                'unmatched_entities': result.unmatched_entities
            },
            'objective_synchronizations': [
                {
                    'objective_id': obj.objective_id,
                    'objective_title': obj.objective_title,
                    'embedding_score': obj.embedding_score,
                    'entity_matches': obj.entity_matches,
                    'entity_score': obj.entity_score,
                    'combined_score': obj.combined_score,
                    'has_strong_support': obj.has_strong_support,
                    'top_matching_actions': obj.top_matching_actions,
                    'gaps': obj.gaps
                }
                for obj in result.objective_synchronizations
            ],
            'strengths': result.strengths,
            'weaknesses': result.weaknesses,
            'recommendations': result.recommendations
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"\n✓ Complete results saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Load analysis results
    with open('embedding_analysis_results.json', 'r') as f:
        embedding_results = json.load(f)
    
    with open('entity_analysis_results.json', 'r') as f:
        entity_results = json.load(f)
    
    with open('strategic_plan.json', 'r') as f:
        strategic_doc = json.load(f)
    
    with open('action_plan.json', 'r') as f:
        action_doc = json.load(f)
    
    # Initialize scoring engine with OpenAI API key
    engine = LLMScoringEngine(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        embedding_weight=0.60,
        entity_weight=0.40,
        strong_support_threshold=75.0
    )
    
    # Combine scores
    final_result = engine.combine_scores(
        embedding_results=embedding_results,
        entity_results=entity_results,
        strategic_doc=strategic_doc,
        action_doc=action_doc
    )
    
    # Save final results
    engine.save_results(final_result, 'final_synchronization_results.json')
    
    print("\n✓ Synchronization assessment complete!")
