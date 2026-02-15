"""
Scoring Engine - Combines Embedding and Entity Analysis
Produces final synchronization score and comprehensive results
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


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


class ScoringEngine:
    """Combines multiple analysis methods into final score"""
    
    def __init__(
        self,
        embedding_weight: float = 0.60,
        entity_weight: float = 0.40,
        strong_support_threshold: float = 75.0
    ):
        """
        Initialize scoring engine
        
        Args:
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
        
        # Generate insights
        strengths = self._identify_strengths(
            objective_syncs,
            embedding_results,
            entity_results
        )
        
        weaknesses = self._identify_weaknesses(
            objective_syncs,
            entity_results
        )
        
        recommendations = self._generate_recommendations(
            objective_syncs,
            weaknesses
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
            
            # Calculate entity score for this objective (simple: % of matches)
            # In a more sophisticated version, this could be weighted
            entity_score = min(entity_match_count * 20, 100)  # 5 matches = 100%
            
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
    
    def _identify_strengths(
        self,
        objective_syncs: List[ObjectiveSynchronization],
        embedding_results: Dict,
        entity_results: Dict
    ) -> List[str]:
        """Identify strengths in the alignment"""
        
        strengths = []
        
        # Check overall alignment
        strong_count = sum(1 for obj in objective_syncs if obj.has_strong_support)
        if strong_count == len(objective_syncs):
            strengths.append("All strategic objectives have strong supporting actions")
        elif strong_count >= len(objective_syncs) * 0.8:
            strengths.append(f"{strong_count}/{len(objective_syncs)} strategic objectives have strong support")
        
        # Check entity matching
        match_rate = entity_results['match_rate']
        if match_rate >= 85:
            strengths.append(f"Excellent entity matching ({match_rate:.0f}%) - KPIs and targets well-aligned")
        elif match_rate >= 70:
            strengths.append(f"Good entity matching ({match_rate:.0f}%) - most targets are tracked")
        
        # Check semantic similarity
        avg_similarity = embedding_results['average_similarity']
        if avg_similarity >= 0.85:
            strengths.append(f"Very high semantic alignment ({avg_similarity:.2f}) - actions clearly address objectives")
        
        # Check for well-aligned objectives
        top_objectives = sorted(objective_syncs, key=lambda x: x.combined_score, reverse=True)[:2]
        if top_objectives and top_objectives[0].combined_score >= 90:
            strengths.append(f"Exemplary alignment on '{top_objectives[0].objective_title}'")
        
        return strengths if strengths else ["Action plan shows general alignment with strategic direction"]
    
    def _identify_weaknesses(
        self,
        objective_syncs: List[ObjectiveSynchronization],
        entity_results: Dict
    ) -> List[str]:
        """Identify weaknesses in the alignment"""
        
        weaknesses = []
        
        # Check for weak objectives
        weak_objectives = [obj for obj in objective_syncs if not obj.has_strong_support]
        if weak_objectives:
            weaknesses.append(f"{len(weak_objectives)} strategic objectives lack strong supporting actions")
        
        # Check for unmatched entities
        unmatched = entity_results.get('unmatched_strategic_entities', [])
        if unmatched:
            # Group by type
            unmatched_by_type = {}
            for entity in unmatched:
                entity_type = entity['type']
                if entity_type not in unmatched_by_type:
                    unmatched_by_type[entity_type] = []
                unmatched_by_type[entity_type].append(entity)
            
            for entity_type, entities in unmatched_by_type.items():
                if len(entities) >= 3:
                    weaknesses.append(f"{len(entities)} {entity_type} entities not tracked in action plan")
        
        # Check match rate
        match_rate = entity_results['match_rate']
        if match_rate < 50:
            weaknesses.append(f"Low entity match rate ({match_rate:.0f}%) - many strategic targets missing from actions")
        
        return weaknesses if weaknesses else []
    
    def _generate_recommendations(
        self,
        objective_syncs: List[ObjectiveSynchronization],
        weaknesses: List[str]
    ) -> List[Dict]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Recommendations for weak objectives
        weak_objectives = sorted(
            [obj for obj in objective_syncs if not obj.has_strong_support],
            key=lambda x: x.combined_score
        )
        
        for obj in weak_objectives[:3]:  # Top 3 weakest
            priority = "high" if obj.combined_score < 50 else "medium"
            
            rec = {
                'priority': priority,
                'objective': obj.objective_title,
                'current_score': obj.combined_score,
                'actions': []
            }
            
            if obj.embedding_score < 70:
                rec['actions'].append(
                    "Review action plan to ensure it directly addresses the strategic intent"
                )
            
            if obj.entity_matches == 0:
                rec['actions'].append(
                    "Add explicit KPIs, targets, and timelines matching strategic plan"
                )
            
            if not obj.top_matching_actions:
                rec['actions'].append(
                    "Create new action items specifically supporting this objective"
                )
            
            recommendations.append(rec)
        
        # General recommendations
        if len(weak_objectives) > 5:
            recommendations.insert(0, {
                'priority': 'high',
                'objective': 'Overall Action Plan',
                'actions': [
                    'Conduct comprehensive review to strengthen objective-action linkages',
                    'Consider adding cross-reference table mapping objectives to actions'
                ]
            })
        
        return recommendations
    
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
        
        print(f"\n✅ Strengths:")
        for strength in result.strengths:
            print(f"   • {strength}")
        
        if result.weaknesses:
            print(f"\n⚠️  Weaknesses:")
            for weakness in result.weaknesses:
                print(f"   • {weakness}")
        
        if result.recommendations:
            print(f"\n💡 Top Recommendations:")
            for i, rec in enumerate(result.recommendations[:3], 1):
                print(f"\n   {i}. [{rec['priority'].upper()}] {rec.get('objective', 'General')}")
                for action in rec.get('actions', []):
                    print(f"      → {action}")
    
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
    
    # Initialize scoring engine
    engine = ScoringEngine(
        embedding_weight=0.60,  # 60% weight
        entity_weight=0.40,     # 40% weight
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
