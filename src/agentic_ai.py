"""
Agentic AI for Strategic Plan Synchronization
Autonomously identifies gaps, generates proposals, and simulates improvements
"""

import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from openai import OpenAI
from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent
# Data folder
DATA_DIR = BASE_DIR / "data"
AGENTIC_AI_RESULTS_PATH = DATA_DIR / "agent_analysis.json"


@dataclass
class CriticalFinding:
    """Represents a critical issue found by the agent"""
    id: str
    severity: str  # "critical", "high", "medium"
    title: str
    description: str
    affected_objective: str
    impact: str
    evidence: List[str]


@dataclass
class ActionProposal:
    """Agent-generated proposal for new action item"""
    id: str
    priority: str  # "high", "medium", "low"
    objective_id: str
    objective_title: str
    action_title: str
    description: str
    budget_estimate: float
    timeline: str
    expected_kpis: List[str]
    rationale: str
    expected_impact: str
    status: str = "pending"  # "pending", "accepted", "rejected"


@dataclass
class ImpactSimulation:
    """Simulated impact of implementing proposals"""
    current_score: float
    projected_score: float
    improvement: float
    affected_objectives: List[Dict]


@dataclass
class AgentAnalysisResult:
    """Complete agent analysis results"""
    timestamp: str
    critical_findings: List[CriticalFinding]
    proposals: List[ActionProposal]
    impact_simulation: ImpactSimulation
    summary: Dict


class AgenticAI:
    """Autonomous AI agent for strategic alignment analysis"""
    
    def __init__(self, openai_api_key: str):
        """
        Initialize Agentic AI
        
        Args:
            openai_api_key: OpenAI API key for GPT-4
        """
        self.openai_client = OpenAI(api_key=openai_api_key)
    
    def analyze(
        self,
        strategic_doc: Dict,
        action_doc: Dict,
        analysis_results: Dict
    ) -> AgentAnalysisResult:
        """
        Run complete autonomous analysis
        
        Args:
            strategic_doc: Strategic plan document
            action_doc: Action plan document
            analysis_results: Synchronization analysis results
            
        Returns:
            AgentAnalysisResult with findings and proposals
        """
        print("\n" + "="*70)
        print("🤖 AGENTIC AI - AUTONOMOUS ANALYSIS")
        print("="*70)
        
        # Step 1: Scan for critical findings
        print("\n[1/4] 🔍 Scanning for critical gaps...")
        critical_findings = self._identify_critical_findings(analysis_results)
        print(f"  Found {len(critical_findings)} critical findings")
        
        # Step 2: Generate action proposals
        print("\n[2/4] 💡 Generating improvement proposals...")
        proposals = self._generate_proposals(
            strategic_doc,
            action_doc,
            analysis_results,
            critical_findings
        )
        print(f"  Generated {len(proposals)} proposals")
        
        # Step 3: Simulate impact
        print("\n[3/4] 📊 Simulating impact of proposals...")
        impact_simulation = self._simulate_impact(
            analysis_results,
            proposals
        )
        print(f"  Projected improvement: {impact_simulation.improvement:.1f}%")
        
        # Step 4: Create summary
        print("\n[4/4] 📝 Generating summary...")
        summary = self._create_summary(
            critical_findings,
            proposals,
            impact_simulation
        )
        
        result = AgentAnalysisResult(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            critical_findings=critical_findings,
            proposals=proposals,
            impact_simulation=impact_simulation,
            summary=summary
        )
        
        print("\n✅ Agent analysis complete!")
        return result
    
    def _identify_critical_findings(
        self,
        analysis_results: Dict
    ) -> List[CriticalFinding]:
        """Identify critical issues requiring immediate attention"""
        
        findings = []
        finding_id = 0
        
        # Check for objectives with very low scores (<50%)
        for obj in analysis_results.get('objective_synchronizations', []):
            if obj['combined_score'] < 50:
                finding_id += 1
                findings.append(CriticalFinding(
                    id=f"critical_{finding_id}",
                    severity="critical",
                    title=f"Severe Misalignment: {obj['objective_title']}",
                    description=f"Objective scoring only {obj['combined_score']:.1f}/100, indicating major gaps in action plan support.",
                    affected_objective=obj['objective_title'],
                    impact="High - This strategic priority lacks adequate execution plan",
                    evidence=[
                        f"Embedding score: {obj['embedding_score']:.1f}%",
                        f"Entity matches: {obj['entity_matches']}",
                        f"Gaps: {', '.join(obj.get('gaps', []))}"
                    ]
                ))
        
        # Check for high-priority objectives with weak support (50-65%)
        for obj in analysis_results.get('objective_synchronizations', []):
            if 50 <= obj['combined_score'] < 65:
                finding_id += 1
                findings.append(CriticalFinding(
                    id=f"high_{finding_id}",
                    severity="high",
                    title=f"Weak Support: {obj['objective_title']}",
                    description=f"Objective scoring {obj['combined_score']:.1f}/100 needs strengthened action support.",
                    affected_objective=obj['objective_title'],
                    impact="Medium-High - Strategic goal at risk of underdelivery",
                    evidence=[
                        f"Only {obj['entity_matches']} entity matches found",
                        *obj.get('gaps', [])[:2]
                    ]
                ))
        
        # Check for missing entity coverage
        if analysis_results.get('summary', {}).get('unmatched_entities', 0) > 10:
            finding_id += 1
            unmatched_count = analysis_results['summary']['unmatched_entities']
            findings.append(CriticalFinding(
                id=f"entities_{finding_id}",
                severity="high",
                title="Significant Entity Coverage Gaps",
                description=f"{unmatched_count} strategic entities (KPIs, targets) not tracked in action plan.",
                affected_objective="Multiple objectives",
                impact="Medium - Accountability and measurement gaps across strategic plan",
                evidence=[
                    f"Total unmatched: {unmatched_count}",
                    f"Match rate: {analysis_results.get('entity_score', 0):.1f}%"
                ]
            ))
        
        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2}
        findings.sort(key=lambda x: severity_order[x.severity])
        
        return findings
    
    def _generate_proposals(
        self,
        strategic_doc: Dict,
        action_doc: Dict,
        analysis_results: Dict,
        critical_findings: List[CriticalFinding]
    ) -> List[ActionProposal]:
        """Generate concrete action proposals using GPT-4"""
        
        proposals = []
        
        # Strategy 1: Get weak objectives that need proposals
        weak_objectives = [
            obj for obj in analysis_results.get('objective_synchronizations', [])
            if obj['combined_score'] < 75
        ]
        
        # Generate proposals for top 3 weakest objectives
        for obj in sorted(weak_objectives, key=lambda x: x['combined_score'])[:3]:
            print(f"  Generating proposals for: {obj['objective_title'][:50]}...")
            
            # Prepare context for LLM
            context = self._build_proposal_context(
                obj,
                strategic_doc,
                action_doc,
                analysis_results
            )
            
            # Generate proposals with GPT-4
            obj_proposals = self._generate_proposals_for_objective(
                obj,
                context
            )
            
            proposals.extend(obj_proposals)
        
        # Strategy 2: If no weak objectives but have critical/high findings, generate proposals
        if len(proposals) == 0 and len(critical_findings) > 0:
            print(f"  No weak objectives, but {len(critical_findings)} critical findings detected")
            print(f"  Generating proposals to address findings...")
            
            # Generate proposals for critical findings
            for finding in critical_findings:
                if finding.severity in ['critical', 'high']:
                    finding_proposals = self._generate_proposals_for_finding(
                        finding,
                        strategic_doc,
                        action_doc,
                        analysis_results
                    )
                    proposals.extend(finding_proposals)
        
        # Strategy 3: Generate entity tracking improvement proposals if entity score is low
        entity_score = analysis_results.get('entity_score', 100)
        if entity_score < 60 and len(proposals) < 3:
            print(f"  Entity match score low ({entity_score:.1f}%), generating tracking improvement proposals...")
            entity_proposals = self._generate_entity_tracking_proposals(
                strategic_doc,
                action_doc,
                analysis_results
            )
            proposals.extend(entity_proposals)
        
        return proposals
    
    def _build_proposal_context(
        self,
        objective: Dict,
        strategic_doc: Dict,
        action_doc: Dict,
        analysis_results: Dict
    ) -> str:
        """Build context for proposal generation"""
        
        # Find strategic objective details
        strategic_section = None
        for section in strategic_doc.get('sections', []):
            if section['title'] == objective['objective_title']:
                strategic_section = section
                break
        
        context = f"""
OBJECTIVE NEEDING IMPROVEMENT:
Title: {objective['objective_title']}
Current Score: {objective['combined_score']:.1f}/100
Embedding Score: {objective['embedding_score']:.1f}/100
Entity Matches: {objective['entity_matches']}

IDENTIFIED GAPS:
{chr(10).join(f"- {gap}" for gap in objective.get('gaps', []))}

STRATEGIC PLAN DETAILS:
"""
        
        if strategic_section:
            context += f"Budget: ${strategic_section.get('budget', 0):,.0f}\n"
            context += f"Timeline: {strategic_section.get('timeline', 'Not specified')}\n"
            
            if strategic_section.get('kpis'):
                context += "\nStrategic KPIs:\n"
                for kpi in strategic_section['kpis'][:5]:
                    kpi_text = kpi.get('metric', 'Unknown')
                    if kpi.get('target'):
                        kpi_text += f": {kpi['target']}{kpi.get('unit', '')}"
                    if kpi.get('deadline'):
                        kpi_text += f" by {kpi['deadline']}"
                    context += f"- {kpi_text}\n"
        
        # Add current matching actions
        if objective.get('top_matching_actions'):
            context += "\nCurrent Best Matching Actions:\n"
            for action in objective['top_matching_actions'][:3]:
                context += f"- {action['action_title']} (similarity: {action['similarity_score']:.2f})\n"
        
        return context
    
    def _generate_proposals_for_finding(
        self,
        finding: CriticalFinding,
        strategic_doc: Dict,
        action_doc: Dict,
        analysis_results: Dict
    ) -> List[ActionProposal]:
        """Generate proposals to address a critical finding"""
        
        print(f"    Generating proposals for finding: {finding.title[:50]}...")
        
        # Build context for the finding
        context = f"""
CRITICAL FINDING TO ADDRESS:
Title: {finding.title}
Severity: {finding.severity.upper()}
Description: {finding.description}
Impact: {finding.impact}

Evidence:
{chr(10).join(f"- {e}" for e in finding.evidence)}

Current State:
- Overall Score: {analysis_results['overall_score']:.1f}/100
- Entity Match Score: {analysis_results.get('entity_score', 0):.1f}%
- Total Unmatched Entities: {analysis_results.get('summary', {}).get('unmatched_entities', 0)}
"""
        
        prompt = f"""You are an expert strategic planning consultant. Based on this critical finding, generate 1-2 SPECIFIC, ACTIONABLE proposals to address the issue.

{context}

The proposals should directly address this finding and improve the alignment. Include:
1. Specific actions to take
2. Clear KPIs to track
3. Realistic budget and timeline
4. Expected measurable impact

Return ONLY valid JSON in this exact format:
{{
  "proposals": [
    {{
      "action_title": "Monthly KPI Tracking Dashboard",
      "description": "Implement comprehensive monthly tracking dashboard for all strategic KPIs with automated alerts for off-track metrics.",
      "budget_estimate": 250000,
      "timeline": "Q1 2025 - Q2 2025",
      "expected_kpis": ["100% KPI coverage", "Monthly tracking reports", "Automated variance alerts"],
      "rationale": "Addresses the 121 unmatched strategic entities by ensuring all KPIs have explicit tracking mechanisms",
      "expected_impact": "Would improve entity match rate from 40.5% to 85%+ by adding systematic tracking for all strategic metrics"
    }}
  ]
}}

Generate 1-2 specific, implementable proposals.
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strategic planning expert who creates specific, actionable proposals. Always return valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.4,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Convert to ActionProposal objects
            proposals = []
            for i, prop in enumerate(result.get('proposals', [])):
                # Use finding severity to determine priority
                if finding.severity == "critical":
                    priority = "high"
                elif finding.severity == "high":
                    priority = "medium"
                else:
                    priority = "low"
                
                # Use first affected objective or "Multiple"
                objective_title = finding.affected_objective
                objective_id = f"finding_{finding.id}"
                
                proposals.append(ActionProposal(
                    id=f"proposal_finding_{finding.id}_{i}",
                    priority=priority,
                    objective_id=objective_id,
                    objective_title=objective_title,
                    action_title=prop['action_title'],
                    description=prop['description'],
                    budget_estimate=prop.get('budget_estimate', 0),
                    timeline=prop.get('timeline', 'TBD'),
                    expected_kpis=prop.get('expected_kpis', []),
                    rationale=prop.get('rationale', ''),
                    expected_impact=prop.get('expected_impact', ''),
                    status='pending'
                ))
            
            print(f"      ✓ Generated {len(proposals)} proposals for finding")
            return proposals
            
        except Exception as e:
            print(f"      ⚠ Proposal generation failed: {e}")
            return []
    
    def _generate_entity_tracking_proposals(
        self,
        strategic_doc: Dict,
        action_doc: Dict,
        analysis_results: Dict
    ) -> List[ActionProposal]:
        """Generate proposals to improve entity tracking and KPI coverage"""
        
        print(f"    Generating entity tracking improvement proposals...")
        
        unmatched_count = analysis_results.get('summary', {}).get('unmatched_entities', 0)
        entity_score = analysis_results.get('entity_score', 0)
        
        # Get sample unmatched entities
        unmatched_entities = analysis_results.get('entity_results', {}).get('unmatched_strategic_entities', [])
        if not unmatched_entities:
            # Fallback - use entity_score to infer problem
            unmatched_entities = []
        
        # Group unmatched by type
        unmatched_by_type = {}
        for entity in unmatched_entities[:20]:  # Top 20
            entity_type = entity.get('type', 'UNKNOWN')
            if entity_type not in unmatched_by_type:
                unmatched_by_type[entity_type] = []
            unmatched_by_type[entity_type].append(entity.get('text', ''))
        
        context = f"""
ENTITY TRACKING GAP ANALYSIS:
- Total Unmatched Strategic Entities: {unmatched_count}
- Entity Match Score: {entity_score:.1f}%
- Overall Synchronization Score: {analysis_results['overall_score']:.1f}/100

Sample Unmatched Entities by Type:
"""
        
        for entity_type, entities in unmatched_by_type.items():
            context += f"\n{entity_type}:\n"
            for entity in entities[:5]:
                context += f"  - {entity}\n"
        
        context += f"""
PROBLEM: Many strategic KPIs, targets, and metrics from the strategic plan are not explicitly tracked or measured in the action plan. This creates accountability gaps and makes it difficult to measure progress toward strategic goals.
"""
        
        prompt = f"""You are an expert strategic planning consultant. Based on this entity tracking gap analysis, generate 2-3 SPECIFIC, ACTIONABLE proposals to improve KPI tracking and measurement.

{context}

Create proposals that:
1. Systematically address the entity tracking gaps
2. Ensure strategic KPIs are measurable in the action plan
3. Create accountability mechanisms
4. Are realistic and implementable

Return ONLY valid JSON in this exact format:
{{
  "proposals": [
    {{
      "action_title": "Strategic KPI Monitoring Framework",
      "description": "Establish comprehensive KPI monitoring framework with monthly dashboards, automated data collection, and variance reporting for all strategic metrics.",
      "budget_estimate": 300000,
      "timeline": "Q1 2025 - Q3 2025",
      "expected_kpis": ["100% strategic KPI coverage", "Monthly KPI dashboards", "Automated alerts for off-track metrics", "Quarterly Board reporting"],
      "rationale": "Addresses systematic gap of {unmatched_count} unmatched entities by creating explicit tracking mechanisms for all strategic metrics",
      "expected_impact": "Would improve entity match score from {entity_score:.1f}% to 85%+ by ensuring every strategic target has a corresponding measurement and tracking mechanism in the action plan"
    }}
  ]
}}

Generate 2-3 specific proposals focused on improving measurement and tracking.
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strategic planning expert specializing in KPI frameworks and performance measurement. Always return valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.4,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Convert to ActionProposal objects
            proposals = []
            for i, prop in enumerate(result.get('proposals', [])):
                proposals.append(ActionProposal(
                    id=f"proposal_entity_tracking_{i}",
                    priority="medium",  # Entity tracking is usually medium priority
                    objective_id="entity_tracking",
                    objective_title="Enterprise-wide KPI Tracking",
                    action_title=prop['action_title'],
                    description=prop['description'],
                    budget_estimate=prop.get('budget_estimate', 0),
                    timeline=prop.get('timeline', 'TBD'),
                    expected_kpis=prop.get('expected_kpis', []),
                    rationale=prop.get('rationale', ''),
                    expected_impact=prop.get('expected_impact', ''),
                    status='pending'
                ))
            
            print(f"      ✓ Generated {len(proposals)} entity tracking proposals")
            return proposals
            
        except Exception as e:
            print(f"      ⚠ Entity tracking proposal generation failed: {e}")
            return []

    def _generate_proposals_for_objective(
        self,
        objective: Dict,
        context: str
    ) -> List[ActionProposal]:  

        """Generate 1-2 specific proposals for an objective using GPT-4"""
        
        prompt = f"""You are an expert strategic planning consultant. Based on the analysis below, generate 1-2 SPECIFIC, ACTIONABLE proposals for new action items to improve alignment.

{context}

Generate concrete proposals that:
1. Address the identified gaps
2. Include specific KPIs to track
3. Have realistic budgets and timelines
4. Can be directly implemented

Return ONLY valid JSON in this exact format:
{{
  "proposals": [
    {{
      "action_title": "Quarterly Risk Assessment Reviews",
      "description": "Implement quarterly comprehensive risk assessment reviews with Board oversight, tracking NPL ratio, tier-1 capital, and credit loss rates against targets.",
      "budget_estimate": 500000,
      "timeline": "Q1 2025 - Q4 2025",
      "expected_kpis": ["NPL ratio <1.5%", "Tier-1 capital >12%", "Credit loss rate <0.8%"],
      "rationale": "Addresses missing timeline milestones and KPI tracking gaps identified in analysis",
      "expected_impact": "Would improve objective score from {objective['combined_score']:.1f} to approximately 78 by adding measurable quarterly milestones and explicit KPI tracking"
    }}
  ]
}}

Generate 1-2 proposals. Be specific with numbers, dates, and KPIs.
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strategic planning expert who creates specific, actionable proposals. Always return valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.4,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Convert to ActionProposal objects
            proposals = []
            for i, prop in enumerate(result.get('proposals', [])):
                # Determine priority based on current score
                if objective['combined_score'] < 50:
                    priority = "high"
                elif objective['combined_score'] < 65:
                    priority = "medium"
                else:
                    priority = "low"
                
                proposals.append(ActionProposal(
                    id=f"proposal_{objective['objective_id']}_{i}",
                    priority=priority,
                    objective_id=objective['objective_id'],
                    objective_title=objective['objective_title'],
                    action_title=prop['action_title'],
                    description=prop['description'],
                    budget_estimate=prop.get('budget_estimate', 0),
                    timeline=prop.get('timeline', 'TBD'),
                    expected_kpis=prop.get('expected_kpis', []),
                    rationale=prop.get('rationale', ''),
                    expected_impact=prop.get('expected_impact', ''),
                    status='pending'
                ))
            
            return proposals
            
        except Exception as e:
            print(f"  ⚠ Proposal generation failed for {objective['objective_title']}: {e}")
            return []
    
    def _simulate_impact(
        self,
        analysis_results: Dict,
        proposals: List[ActionProposal]
    ) -> ImpactSimulation:
        """Simulate impact of implementing proposals"""
        
        current_score = analysis_results['overall_score']
        
        # Simple heuristic: each proposal improves its objective by ~10-15 points
        improvements_by_objective = {}
        
        # Track entity tracking improvements separately
        entity_tracking_improvement = 0
        
        for proposal in proposals:
            obj_id = proposal.objective_id
            
            # Special handling for entity tracking proposals
            if obj_id == "entity_tracking" or "entity" in obj_id.lower() or "finding" in obj_id:
                # Entity tracking proposals improve overall entity score
                entity_tracking_improvement += 8.0  # Each proposal improves entity score
            else:
                # Regular objective proposals
                if obj_id not in improvements_by_objective:
                    improvements_by_objective[obj_id] = 0
                
                # Estimate improvement (diminishing returns)
                base_improvement = 12.0
                existing_improvements = improvements_by_objective[obj_id] / base_improvement
                diminishing_factor = 0.7 ** existing_improvements
                
                improvements_by_objective[obj_id] += base_improvement * diminishing_factor
        
        # Calculate projected objective scores
        affected_objectives = []
        total_improvement = 0
        
        for obj in analysis_results.get('objective_synchronizations', []):
            if obj['objective_id'] in improvements_by_objective:
                improvement = improvements_by_objective[obj['objective_id']]
                current = obj['combined_score']
                projected = min(current + improvement, 100)
                
                affected_objectives.append({
                    'objective_title': obj['objective_title'],
                    'current_score': current,
                    'projected_score': projected,
                    'improvement': projected - current
                })
                
                total_improvement += (projected - current)
        
        # Add entity tracking as an affected "objective" if we have those proposals
        if entity_tracking_improvement > 0:
            entity_current = analysis_results.get('entity_score', 0)
            entity_projected = min(entity_current + entity_tracking_improvement, 100)
            
            affected_objectives.append({
                'objective_title': 'Entity Tracking & KPI Coverage',
                'current_score': entity_current,
                'projected_score': entity_projected,
                'improvement': entity_projected - entity_current
            })
            
            # Entity tracking improvements contribute to overall score
            # Entity score is 40% of overall (based on weights)
            entity_contribution = (entity_projected - entity_current) * 0.40
            total_improvement += entity_contribution
        
        # Calculate overall score improvement
        if affected_objectives:
            # If we have objective improvements, use them
            if improvements_by_objective:
                num_objectives = len(analysis_results.get('objective_synchronizations', []))
                avg_improvement = total_improvement / num_objectives if num_objectives > 0 else total_improvement
                projected_overall = min(current_score + avg_improvement, 100)
            else:
                # Only entity tracking improvements
                projected_overall = min(current_score + entity_tracking_improvement * 0.40, 100)
        else:
            projected_overall = current_score
        
        return ImpactSimulation(
            current_score=current_score,
            projected_score=projected_overall,
            improvement=projected_overall - current_score,
            affected_objectives=affected_objectives
        )
    
    def _create_summary(
        self,
        critical_findings: List[CriticalFinding],
        proposals: List[ActionProposal],
        impact_simulation: ImpactSimulation
    ) -> Dict:
        """Create executive summary of agent analysis"""
        
        return {
            'total_findings': len(critical_findings),
            'critical_count': len([f for f in critical_findings if f.severity == "critical"]),
            'high_count': len([f for f in critical_findings if f.severity == "high"]),
            'total_proposals': len(proposals),
            'high_priority_proposals': len([p for p in proposals if p.priority == "high"]),
            'current_score': impact_simulation.current_score,
            'projected_score': impact_simulation.projected_score,
            'improvement': impact_simulation.improvement,
            'objectives_affected': len(impact_simulation.affected_objectives)
        }
    
    def save_results(self, result: AgentAnalysisResult, output_path: str):
        """Save agent analysis results to JSON"""
        
        # Convert to dict
        result_dict = {
            'timestamp': result.timestamp,
            'critical_findings': [
                {
                    'id': f.id,
                    'severity': f.severity,
                    'title': f.title,
                    'description': f.description,
                    'affected_objective': f.affected_objective,
                    'impact': f.impact,
                    'evidence': f.evidence
                }
                for f in result.critical_findings
            ],
            'proposals': [
                {
                    'id': p.id,
                    'priority': p.priority,
                    'objective_id': p.objective_id,
                    'objective_title': p.objective_title,
                    'action_title': p.action_title,
                    'description': p.description,
                    'budget_estimate': p.budget_estimate,
                    'timeline': p.timeline,
                    'expected_kpis': p.expected_kpis,
                    'rationale': p.rationale,
                    'expected_impact': p.expected_impact,
                    'status': p.status
                }
                for p in result.proposals
            ],
            'impact_simulation': {
                'current_score': result.impact_simulation.current_score,
                'projected_score': result.impact_simulation.projected_score,
                'improvement': result.impact_simulation.improvement,
                'affected_objectives': result.impact_simulation.affected_objectives
            },
            'summary': result.summary
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"\n✓ Agent results saved to {output_path}")
    
    def accept_proposal(
        self,
        proposal_id: str,
        action_doc: Dict,
        output_path: str
    ) -> Dict:
        """
        Accept a proposal and add it to the action plan
        
        Args:
            proposal_id: ID of proposal to accept
            action_doc: Current action plan document
            output_path: Path to save updated action plan
            
        Returns:
            Updated action plan document
        """
        # Load agent results
        with open(AGENTIC_AI_RESULTS_PATH, 'r') as f:
            agent_results = json.load(f)
        
        # Find the proposal
        proposal = None
        for p in agent_results['proposals']:
            if p['id'] == proposal_id:
                proposal = p
                break
        
        if not proposal:
            raise ValueError(f"Proposal {proposal_id} not found")
        
        # Create new action section
        new_action_id = f"action_agent_{len(action_doc['sections']) + 1}"
        
        new_section = {
            'id': new_action_id,
            'type': 'action_item',
            'title': proposal['action_title'],
            'content': proposal['description'],
            'kpis': [
                {'metric': kpi, 'target': None, 'unit': '', 'deadline': None}
                for kpi in proposal['expected_kpis']
            ],
            'budget': proposal['budget_estimate'],
            'timeline': proposal['timeline'],
            'initiatives': [],
            'priority': proposal['priority']
        }
        
        # Add to action plan
        action_doc['sections'].append(new_section)
        
        # Update total budget
        if action_doc.get('total_budget'):
            action_doc['total_budget'] += proposal['budget_estimate']
        
        # Mark proposal as accepted
        proposal['status'] = 'accepted'
        
        # Save updated documents
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(action_doc, f, indent=2)
        
        with open(AGENTIC_AI_RESULTS_PATH, 'w', encoding='utf-8') as f:
            json.dump(agent_results, f, indent=2)
        
        print(f"✓ Proposal accepted and added to action plan: {proposal['action_title']}")
        
        return action_doc

