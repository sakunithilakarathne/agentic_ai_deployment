"""
Entity Extractor for Strategic Plan Synchronization
Extracts and matches explicit entities (KPIs, budgets, timelines, goals)
"""

import re
import json
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import spacy
from fuzzywuzzy import fuzz


@dataclass
class Entity:
    """Represents an extracted entity"""
    text: str
    type: str  # KPI, BUDGET, TIMELINE, GOAL, INITIATIVE, METRIC_TARGET
    value: Optional[str] = None
    source_section: str = ""
    source_title: str = ""


@dataclass
class EntityMatch:
    """Represents a match between strategic and action entities"""
    strategic_entity: Entity
    action_entity: Entity
    match_score: float  # 0-100
    match_type: str  # "exact", "fuzzy", "partial"


@dataclass
class EntityAnalysisResult:
    """Complete entity matching analysis results"""
    overall_score: float  # 0-100
    total_strategic_entities: int
    matched_entities: int
    unmatched_entities: int
    match_rate: float  # percentage
    matches_by_type: Dict[str, int]
    entity_matches: List[EntityMatch]
    unmatched_strategic_entities: List[Entity]
    strategic_entities: Dict[str, List[Entity]]
    action_entities: Dict[str, List[Entity]]


class EntityExtractor:
    """Extracts and matches entities from documents"""
    
    def __init__(self, fuzzy_threshold: int = 85):
        """
        Initialize entity extractor
        
        Args:
            fuzzy_threshold: Minimum fuzzy match score (0-100) to consider a match
        """
        self.fuzzy_threshold = fuzzy_threshold
        
        # Entity type weights (for scoring)
        self.entity_weights = {
            'METRIC_TARGET': 3.0,  # Highest priority (e.g., "75% digital adoption")
            'KPI': 3.0,
            'BUDGET': 2.5,
            'TIMELINE': 2.0,
            'GOAL': 1.5,
            'INITIATIVE': 1.5
        }
        
        # Load spaCy for NLP
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            print("⚠ spaCy model not found. Installing...")
            import os
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
    
    def extract_kpis(self, text: str, section_id: str, section_title: str) -> List[Entity]:
        """Extract KPI names"""
        kpis = []
        
        # Common KPI patterns
        kpi_patterns = [
            r'(?:KPI|metric|indicator):\s*([A-Z][A-Za-z\s-]+)',
            r'\b([A-Z][A-Za-z\s-]+?)\s+(?:rate|ratio|score|index)',
            r'(?:improve|increase|reduce|achieve)\s+([A-Za-z\s-]+?)\s+(?:from|to|by)',
        ]
        
        for pattern in kpi_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                kpi_name = match.group(1).strip()
                
                # Filter out common false positives
                if len(kpi_name) > 5 and len(kpi_name) < 50:
                    if not any(word in kpi_name.lower() for word in ['the', 'this', 'that', 'with']):
                        kpis.append(Entity(
                            text=kpi_name,
                            type='KPI',
                            source_section=section_id,
                            source_title=section_title
                        ))
        
        # Extract from explicit KPI lists
        kpi_section = re.search(r'Key Performance Indicators?.*?:(.*?)(?:\n\n|\n###)', text, re.IGNORECASE | re.DOTALL)
        if kpi_section:
            kpi_text = kpi_section.group(1)
            # Find bullet points or numbered items
            items = re.findall(r'[-•*\d+\.]\s*([A-Z][^\n]+)', kpi_text)
            for item in items:
                clean_item = re.sub(r'\(.*?\)', '', item).strip()
                if len(clean_item) > 5:
                    kpis.append(Entity(
                        text=clean_item,
                        type='KPI',
                        source_section=section_id,
                        source_title=section_title
                    ))
        
        return self._deduplicate_entities(kpis)
    
    def extract_metric_targets(self, text: str, section_id: str, section_title: str) -> List[Entity]:
        """Extract specific metric targets (e.g., '75% digital adoption')"""
        targets = []
        
        # Pattern 1: "X% metric by date"
        pattern1 = r'(\d+(?:\.\d+)?%)\s+([A-Za-z][A-Za-z\s-]+?)\s+(?:by|in)\s+(Q\d\s+\d{4}|\d{4})'
        matches1 = re.finditer(pattern1, text, re.IGNORECASE)
        for match in matches1:
            target_text = f"{match.group(1)} {match.group(2)} by {match.group(3)}"
            targets.append(Entity(
                text=target_text,
                type='METRIC_TARGET',
                value=match.group(1),
                source_section=section_id,
                source_title=section_title
            ))
        
        # Pattern 2: "metric from X to Y"
        pattern2 = r'([A-Za-z][A-Za-z\s-]+?)\s+from\s+(\d+(?:\.\d+)?%?)\s+to\s+(\d+(?:\.\d+)?%?)'
        matches2 = re.finditer(pattern2, text, re.IGNORECASE)
        for match in matches2:
            target_text = f"{match.group(1)} from {match.group(2)} to {match.group(3)}"
            targets.append(Entity(
                text=target_text,
                type='METRIC_TARGET',
                value=match.group(3),
                source_section=section_id,
                source_title=section_title
            ))
        
        # Pattern 3: "target: X"
        pattern3 = r'(?:target|goal):\s*(\d+(?:\.\d+)?%?[A-Z]*)\b'
        matches3 = re.finditer(pattern3, text, re.IGNORECASE)
        for match in matches3:
            # Find preceding metric name
            context_start = max(0, match.start() - 100)
            context = text[context_start:match.start()]
            metric_match = re.search(r'([A-Z][A-Za-z\s-]+?)(?:\s*:|\s*\()', context)
            
            if metric_match:
                metric_name = metric_match.group(1).strip()
                target_text = f"{metric_name}: {match.group(1)}"
                targets.append(Entity(
                    text=target_text,
                    type='METRIC_TARGET',
                    value=match.group(1),
                    source_section=section_id,
                    source_title=section_title
                ))
        
        return self._deduplicate_entities(targets)
    
    def extract_budgets(self, text: str, section_id: str, section_title: str) -> List[Entity]:
        """Extract budget amounts"""
        budgets = []
        
        # Pattern: $XX.XM, $XXM, $X.XB, etc.
        pattern = r'\$\s*([\d,]+(?:\.\d+)?)\s*(M|million|B|billion)?'
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            budget_text = match.group(0)
            amount_str = match.group(1).replace(',', '')
            amount = float(amount_str)
            unit = match.group(2)
            
            if unit:
                unit_lower = unit.lower()
                if unit_lower in ['m', 'million']:
                    amount *= 1_000_000
                elif unit_lower in ['b', 'billion']:
                    amount *= 1_000_000_000
            
            budgets.append(Entity(
                text=budget_text,
                type='BUDGET',
                value=str(int(amount)),
                source_section=section_id,
                source_title=section_title
            ))
        
        return budgets
    
    def extract_timelines(self, text: str, section_id: str, section_title: str) -> List[Entity]:
        """Extract timeline information"""
        timelines = []
        
        # Pattern 1: Q1 2025 - Q4 2026
        pattern1 = r'(Q\d\s+\d{4})\s*[-–—]\s*(Q\d\s+\d{4})'
        matches1 = re.finditer(pattern1, text)
        for match in matches1:
            timeline_text = f"{match.group(1)} - {match.group(2)}"
            timelines.append(Entity(
                text=timeline_text,
                type='TIMELINE',
                value=timeline_text,
                source_section=section_id,
                source_title=section_title
            ))
        
        # Pattern 2: by Q4 2027
        pattern2 = r'by\s+(Q\d\s+\d{4})'
        matches2 = re.finditer(pattern2, text, re.IGNORECASE)
        for match in matches2:
            timeline_text = match.group(1)
            timelines.append(Entity(
                text=timeline_text,
                type='TIMELINE',
                value=timeline_text,
                source_section=section_id,
                source_title=section_title
            ))
        
        # Pattern 3: 2025-2028
        pattern3 = r'\b(20\d{2})\s*[-–—]\s*(20\d{2})\b'
        matches3 = re.finditer(pattern3, text)
        for match in matches3:
            timeline_text = f"{match.group(1)}-{match.group(2)}"
            timelines.append(Entity(
                text=timeline_text,
                type='TIMELINE',
                value=timeline_text,
                source_section=section_id,
                source_title=section_title
            ))
        
        return self._deduplicate_entities(timelines)
    
    def extract_goals(self, text: str, section_id: str, section_title: str) -> List[Entity]:
        """Extract strategic goals"""
        goals = []
        
        # Look for goal-related phrases
        goal_patterns = [
            r'(?:Goal|Objective|Aim):\s*([^\n]+)',
            r'(?:achieve|accomplish|reach)\s+([A-Za-z][^\n.!?]{10,100})',
            r'(?:transform|improve|increase|reduce|enhance)\s+([A-Za-z][^\n.!?]{10,100})',
        ]
        
        for pattern in goal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                goal_text = match.group(1).strip()
                
                # Clean up
                goal_text = re.sub(r'\s+', ' ', goal_text)
                
                if len(goal_text) > 10 and len(goal_text) < 150:
                    goals.append(Entity(
                        text=goal_text,
                        type='GOAL',
                        source_section=section_id,
                        source_title=section_title
                    ))
        
        return self._deduplicate_entities(goals)
    
    def extract_initiatives(self, text: str, section_id: str, section_title: str) -> List[Entity]:
        """Extract initiative names"""
        initiatives = []
        
        # Pattern 1: Bold initiative names (from markdown)
        pattern1 = r'\*\*([A-Z][^*\n]{5,80})\*\*'
        matches1 = re.finditer(pattern1, text)
        for match in matches1:
            initiative_text = match.group(1).strip()
            if ':' not in initiative_text:  # Avoid headers
                initiatives.append(Entity(
                    text=initiative_text,
                    type='INITIATIVE',
                    source_section=section_id,
                    source_title=section_title
                ))
        
        # Pattern 2: "Initiative:" followed by name
        pattern2 = r'(?:Initiative|Project):\s*([A-Z][^\n.]{5,80})'
        matches2 = re.finditer(pattern2, text, re.IGNORECASE)
        for match in matches2:
            initiative_text = match.group(1).strip()
            initiatives.append(Entity(
                text=initiative_text,
                type='INITIATIVE',
                source_section=section_id,
                source_title=section_title
            ))
        
        # Pattern 3: Numbered list items that look like initiatives
        pattern3 = r'\d+\.\s+([A-Z][A-Za-z\s&-]{5,60})(?:\n|:)'
        matches3 = re.finditer(pattern3, text)
        for match in matches3:
            initiative_text = match.group(1).strip()
            if not any(word in initiative_text.lower() for word in ['section', 'chapter', 'appendix']):
                initiatives.append(Entity(
                    text=initiative_text,
                    type='INITIATIVE',
                    source_section=section_id,
                    source_title=section_title
                ))
        
        return self._deduplicate_entities(initiatives)
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities"""
        seen = set()
        unique = []
        
        for entity in entities:
            # Normalize for comparison
            normalized = entity.text.lower().strip()
            if normalized not in seen and len(normalized) > 3:
                seen.add(normalized)
                unique.append(entity)
        
        return unique
    
    def extract_all_entities(self, document: Dict, doc_type: str) -> Dict[str, List[Entity]]:
        """
        Extract all entities from a document
        
        Args:
            document: Document dict from document_processor
            doc_type: 'strategic_plan' or 'action_plan'
            
        Returns:
            Dict mapping entity type to list of entities
        """
        print(f"\nExtracting entities from {doc_type}...")
        
        all_entities = defaultdict(list)
        
        for section in document['sections']:
            section_id = section['id']
            section_title = section['title']
            content = section['content']
            
            print(f"  Processing: {section_title[:50]}...")
            
            # Extract each entity type
            all_entities['KPI'].extend(
                self.extract_kpis(content, section_id, section_title)
            )
            all_entities['METRIC_TARGET'].extend(
                self.extract_metric_targets(content, section_id, section_title)
            )
            all_entities['BUDGET'].extend(
                self.extract_budgets(content, section_id, section_title)
            )
            all_entities['TIMELINE'].extend(
                self.extract_timelines(content, section_id, section_title)
            )
            all_entities['GOAL'].extend(
                self.extract_goals(content, section_id, section_title)
            )
            all_entities['INITIATIVE'].extend(
                self.extract_initiatives(content, section_id, section_title)
            )
        
        # Print summary
        for entity_type, entities in all_entities.items():
            print(f"    {entity_type}: {len(entities)} found")
        
        return dict(all_entities)
    
    def fuzzy_match(self, text1: str, text2: str) -> Tuple[int, str]:
        """
        Calculate fuzzy match score between two strings
        
        Returns:
            (score, match_type) where score is 0-100
        """
        text1_lower = text1.lower().strip()
        text2_lower = text2.lower().strip()
        
        # Exact match
        if text1_lower == text2_lower:
            return 100, "exact"
        
        # Token sort ratio (handles word order differences)
        token_score = fuzz.token_sort_ratio(text1_lower, text2_lower)
        
        # Partial ratio (handles substring matches)
        partial_score = fuzz.partial_ratio(text1_lower, text2_lower)
        
        # Use the higher score
        score = max(token_score, partial_score)
        
        if score >= 95:
            match_type = "exact"
        elif score >= self.fuzzy_threshold:
            match_type = "fuzzy"
        elif score >= 60:
            match_type = "partial"
        else:
            match_type = "no_match"
        
        return score, match_type
    
    def match_entities(
        self,
        strategic_entities: Dict[str, List[Entity]],
        action_entities: Dict[str, List[Entity]]
    ) -> List[EntityMatch]:
        """
        Match strategic entities with action entities
        
        Returns:
            List of EntityMatch objects
        """
        print("\n" + "="*60)
        print("MATCHING ENTITIES")
        print("="*60)
        
        all_matches = []
        
        for entity_type in strategic_entities.keys():
            if entity_type not in action_entities:
                continue
            
            print(f"\nMatching {entity_type} entities...")
            
            sp_entities = strategic_entities[entity_type]
            ap_entities = action_entities[entity_type]
            
            for sp_entity in sp_entities:
                best_match = None
                best_score = 0
                best_type = "no_match"
                
                for ap_entity in ap_entities:
                    score, match_type = self.fuzzy_match(sp_entity.text, ap_entity.text)
                    
                    if score > best_score:
                        best_score = score
                        best_match = ap_entity
                        best_type = match_type
                
                if best_score >= self.fuzzy_threshold:
                    match = EntityMatch(
                        strategic_entity=sp_entity,
                        action_entity=best_match,
                        match_score=best_score,
                        match_type=best_type
                    )
                    all_matches.append(match)
                    
                    print(f"  ✓ Matched ({best_score}): {sp_entity.text[:50]}...")
        
        return all_matches
    
    def calculate_entity_score(
        self,
        strategic_entities: Dict[str, List[Entity]],
        action_entities: Dict[str, List[Entity]],
        matches: List[EntityMatch]
    ) -> EntityAnalysisResult:
        """
        Calculate overall entity matching score
        
        Returns:
            EntityAnalysisResult with scores and analysis
        """
        print("\n" + "="*60)
        print("CALCULATING ENTITY MATCHING SCORE")
        print("="*60)
        
        # Calculate weighted total
        total_weighted_entities = 0
        matched_weighted_entities = 0
        
        matches_by_type = defaultdict(int)
        matched_entity_ids = set()
        
        for match in matches:
            matches_by_type[match.strategic_entity.type] += 1
            matched_entity_ids.add(match.strategic_entity.text.lower())
        
        # Calculate weighted score
        for entity_type, entities in strategic_entities.items():
            weight = self.entity_weights.get(entity_type, 1.0)
            
            for entity in entities:
                total_weighted_entities += weight
                
                if entity.text.lower() in matched_entity_ids:
                    matched_weighted_entities += weight
        
        # Calculate metrics
        match_rate = (matched_weighted_entities / total_weighted_entities * 100) if total_weighted_entities > 0 else 0
        
        total_sp_entities = sum(len(entities) for entities in strategic_entities.values())
        matched_count = len(matched_entity_ids)
        unmatched_count = total_sp_entities - matched_count
        
        # Find unmatched entities
        unmatched = []
        for entity_type, entities in strategic_entities.items():
            for entity in entities:
                if entity.text.lower() not in matched_entity_ids:
                    unmatched.append(entity)
        
        result = EntityAnalysisResult(
            overall_score=match_rate,
            total_strategic_entities=total_sp_entities,
            matched_entities=matched_count,
            unmatched_entities=unmatched_count,
            match_rate=match_rate,
            matches_by_type=dict(matches_by_type),
            entity_matches=matches,
            unmatched_strategic_entities=unmatched,
            strategic_entities=strategic_entities,
            action_entities=action_entities
        )
        
        self._print_summary(result)
        
        return result
    
    def _print_summary(self, result: EntityAnalysisResult):
        """Print analysis summary"""
        print(f"\nEntity Matching Score: {result.overall_score:.1f}/100")
        print(f"Match Rate: {result.match_rate:.1f}%")
        print(f"\nTotal Strategic Entities: {result.total_strategic_entities}")
        print(f"Matched: {result.matched_entities}")
        print(f"Unmatched: {result.unmatched_entities}")
        
        print("\nMatches by Type:")
        for entity_type, count in result.matches_by_type.items():
            total = len(result.strategic_entities.get(entity_type, []))
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {entity_type}: {count}/{total} ({percentage:.0f}%)")
        
        if result.unmatched_strategic_entities:
            print(f"\n⚠ Top 5 Unmatched Strategic Entities:")
            for entity in result.unmatched_strategic_entities[:5]:
                print(f"  - [{entity.type}] {entity.text[:60]}...")
    
    def analyze_documents(
        self,
        strategic_doc: Dict,
        action_doc: Dict
    ) -> EntityAnalysisResult:
        """
        Complete entity matching analysis
        
        Args:
            strategic_doc: Strategic plan document dict
            action_doc: Action plan document dict
            
        Returns:
            EntityAnalysisResult with complete analysis
        """
        print("\n" + "="*60)
        print("ENTITY MATCHING ANALYSIS")
        print("="*60)
        
        # Extract entities from both documents
        strategic_entities = self.extract_all_entities(strategic_doc, 'strategic_plan')
        action_entities = self.extract_all_entities(action_doc, 'action_plan')
        
        # Match entities
        matches = self.match_entities(strategic_entities, action_entities)
        
        # Calculate score
        result = self.calculate_entity_score(
            strategic_entities,
            action_entities,
            matches
        )
        
        return result
    
    def save_results(self, result: EntityAnalysisResult, output_path: str):
        """Save results to JSON file"""
        result_dict = {
            'overall_score': result.overall_score,
            'total_strategic_entities': result.total_strategic_entities,
            'matched_entities': result.matched_entities,
            'unmatched_entities': result.unmatched_entities,
            'match_rate': result.match_rate,
            'matches_by_type': result.matches_by_type,
            'entity_matches': [
                {
                    'strategic_entity': {
                        'text': m.strategic_entity.text,
                        'type': m.strategic_entity.type,
                        'source': m.strategic_entity.source_title
                    },
                    'action_entity': {
                        'text': m.action_entity.text,
                        'type': m.action_entity.type,
                        'source': m.action_entity.source_title
                    },
                    'match_score': m.match_score,
                    'match_type': m.match_type
                } for m in result.entity_matches
            ],
            'unmatched_strategic_entities': [
                {
                    'text': e.text,
                    'type': e.type,
                    'source': e.source_title
                } for e in result.unmatched_strategic_entities
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"\n✓ Results saved to {output_path}")


# Example usage
if __name__ == "__main__":
    import json
    
    # Initialize extractor
    extractor = EntityExtractor(fuzzy_threshold=85)
    
    # Load documents
    with open('strategic_plan.json', 'r') as f:
        strategic_doc = json.load(f)
    
    with open('action_plan.json', 'r') as f:
        action_doc = json.load(f)
    
    # Analyze
    result = extractor.analyze_documents(strategic_doc, action_doc)
    
    # Save results
    extractor.save_results(result, 'entity_analysis_results.json')
    
    print("\n✓ Entity matching analysis complete!")
