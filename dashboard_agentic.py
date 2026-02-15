"""
Enhanced Dashboard with Agentic AI Integration
Includes manual agent run, caching, re-analysis, and proposal acceptance
"""

import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os
from dotenv import load_dotenv
from pathlib import Path
import sys


# Project root
BASE_DIR = Path(__file__).resolve().parent
# Data folder
DATA_DIR = BASE_DIR / "data"

STRATEGIC_PLAN_PATH = DATA_DIR / "strategic_plan.json"
ACTION_PLAN_PATH = DATA_DIR / "action_plan.json"
LLM_SYNCHRONIZATION_RESULTS_PATH = DATA_DIR / "llm_synchronization_results.json"
AGENTIC_AI_RESULTS_PATH = DATA_DIR / "agent_analysis.json"


# Import modules
try:
    from src.rag_pipeline import RAGPipeline
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

try:
    from src.agentic_ai import AgenticAI
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False


# Page configuration
st.set_page_config(
    page_title="Strategic Plan Synchronization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .big-score {
        font-size: 72px;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
    }
    .good-score { color: #28a745; }
    .medium-score { color: #ffc107; }
    .poor-score { color: #dc3545; }
    .qa-question {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
        color: #1565c0;
        font-weight: 500;
    }
    .qa-answer {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #e0e0e0;
        color: #000000;
    }
    .finding-critical {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .finding-high {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .proposal-card {
        background-color: #f5f5f5;
        border-left: 4px solid #2196f3;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_results():
    """Load analysis results"""
    try:
        with open(LLM_SYNCHRONIZATION_RESULTS_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


@st.cache_resource
def initialize_rag():
    """Initialize RAG pipeline"""
    if not RAG_AVAILABLE:
        return None
    
    load_dotenv()
    openai_key = os.getenv('OPENAI_API_KEY')
    pinecone_key = os.getenv('PINECONE_API_KEY')
    
    if not openai_key or not pinecone_key:
        return None
    
    try:
        return RAGPipeline(openai_key, pinecone_key, "strategic-rag")
    except:
        return None


def load_agent_results():
    """Load cached agent analysis results"""
    try:
        with open(AGENTIC_AI_RESULTS_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def run_agent_analysis():
    """Run the agentic AI analysis"""
    load_dotenv()
    
    # Load documents
    with open(STRATEGIC_PLAN_PATH, 'r') as f:
        strategic_doc = json.load(f)
    
    with open(ACTION_PLAN_PATH, 'r') as f:
        action_doc = json.load(f)
    
    with open(LLM_SYNCHRONIZATION_RESULTS_PATH, 'r') as f:
        analysis_results = json.load(f)
    
    # Run agent
    agent = AgenticAI(openai_api_key=os.getenv('OPENAI_API_KEY'))
    result = agent.analyze(strategic_doc, action_doc, analysis_results)
    agent.save_results(result, AGENTIC_AI_RESULTS_PATH)
    
    return json.loads(open(AGENTIC_AI_RESULTS_PATH, 'r').read())


def accept_proposal(proposal_id: str):
    """Accept a proposal and add to action plan"""
    load_dotenv()
    
    # Load current action plan
    with open(ACTION_PLAN_PATH, 'r') as f:
        action_doc = json.load(f)
    
    # Accept proposal
    agent = AgenticAI(openai_api_key=os.getenv('OPENAI_API_KEY'))
    updated_action_doc = agent.accept_proposal(
        proposal_id=proposal_id,
        action_doc=action_doc,
        output_path=ACTION_PLAN_PATH
    )
    
    return True


def render_agent_page():
    """Render Agentic AI analysis page"""
    st.header("🤖 Autonomous AI Agent Analysis")
    
    if not AGENT_AVAILABLE:
        st.error("Agentic AI module not available. Check installation.")
        return
    
    # Check for cached results
    agent_results = load_agent_results()
    
    if agent_results:
        # Show cached results with timestamp
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.info(f"📅 Last analysis: {agent_results['timestamp']}")
        
        with col2:
            if st.button("🔄 Re-analyze", type="secondary", use_container_width=True):
                with st.spinner("🤖 Agent re-analyzing... (30-60 seconds)"):
                    agent_results = run_agent_analysis()
                    st.success("✅ Analysis complete!")
                    st.rerun()
        
        with col3:
            st.metric("Findings", agent_results['summary']['total_findings'])
        
        # Display results
        display_agent_results(agent_results)
    
    else:
        # No cached results - prompt to run
        st.warning("🤖 **Agent analysis not yet run for this assessment.**")
        
        st.info("""
        The AI Agent will autonomously:
        - 🔍 Scan for critical gaps and misalignments
        - 💡 Generate concrete action proposals
        - 📊 Simulate projected impact of improvements
        - 🎯 Prioritize recommendations by urgency
        
        This may take 30-60 seconds.
        """)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🚀 Run AI Agent Analysis", type="primary", use_container_width=True):
                with st.spinner("🤖 Agent analyzing... Please wait..."):
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("🔍 Scanning for critical gaps...")
                    progress_bar.progress(25)
                    
                    # Run analysis
                    agent_results = run_agent_analysis()
                    
                    status_text.text("💡 Generating proposals...")
                    progress_bar.progress(75)
                    
                    status_text.text("✅ Complete!")
                    progress_bar.progress(100)
                    
                    st.success("Agent analysis complete!")
                    st.rerun()


def display_agent_results(results):
    """Display agent analysis results"""
    
    # Summary metrics
    st.markdown("---")
    st.subheader("📊 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Findings",
            results['summary']['total_findings'],
            help="Critical issues identified by agent"
        )
    
    with col2:
        st.metric(
            "Proposals Generated",
            results['summary']['total_proposals'],
            help="Agent-generated action proposals"
        )
    
    with col3:
        current = results['summary']['current_score']
        projected = results['summary']['projected_score']
        st.metric(
            "Current Score",
            f"{current:.1f}/100",
            f"+{projected - current:.1f} projected",
            delta_color="normal"
        )
    
    with col4:
        improvement = results['summary']['improvement']
        st.metric(
            "Potential Gain",
            f"+{improvement:.1f}%",
            help="If all proposals implemented"
        )
    
    # Critical Findings
    st.markdown("---")
    st.subheader("🔴 Critical Findings")
    
    if results.get('critical_findings'):
        for finding in results['critical_findings']:
            if finding['severity'] == 'critical':
                css_class = 'finding-critical'
                icon = '🔴'
            else:
                css_class = 'finding-high'
                icon = '🟡'
            
            with st.expander(f"{icon} {finding['title']}", expanded=(finding['severity']=='critical')):
                st.markdown(f"**Affected:** {finding['affected_objective']}")
                st.markdown(f"**Impact:** {finding['impact']}")
                st.write(finding['description'])
                
                if finding.get('evidence'):
                    st.write("**Evidence:**")
                    for evidence in finding['evidence']:
                        st.write(f"• {evidence}")
    else:
        st.success("✅ No critical findings - alignment is strong!")
    
    # AI-Generated Proposals
    st.markdown("---")
    st.subheader("💡 AI-Generated Action Proposals")
    
    if results.get('proposals'):
        # Filter proposals by status
        pending_proposals = [p for p in results['proposals'] if p.get('status', 'pending') == 'pending']
        accepted_proposals = [p for p in results['proposals'] if p.get('status') == 'accepted']
        
        # Tabs for pending vs accepted
        tab1, tab2 = st.tabs([f"⏳ Pending ({len(pending_proposals)})", f"✅ Accepted ({len(accepted_proposals)})"])
        
        with tab1:
            if pending_proposals:
                for i, proposal in enumerate(pending_proposals):
                    render_proposal_card(proposal, i, is_pending=True)
            else:
                st.info("No pending proposals")
        
        with tab2:
            if accepted_proposals:
                for i, proposal in enumerate(accepted_proposals):
                    render_proposal_card(proposal, i, is_pending=False)
            else:
                st.info("No accepted proposals yet")
    else:
        st.info("No proposals generated")
    
    # Impact Simulation
    st.markdown("---")
    st.subheader("📊 Impact Simulation")
    
    if results.get('impact_simulation'):
        impact = results['impact_simulation']
        
        # Current vs Projected chart
        col1, col2 = st.columns(2)
        
        with col1:
            # Overall score comparison
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Current',
                x=['Overall Score'],
                y=[impact['current_score']],
                marker_color='#ff9800'
            ))
            
            fig.add_trace(go.Bar(
                name='Projected',
                x=['Overall Score'],
                y=[impact['projected_score']],
                marker_color='#4caf50'
            ))
            
            fig.update_layout(
                title="Overall Score: Current vs Projected",
                yaxis_title="Score",
                yaxis_range=[0, 100],
                barmode='group',
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Affected objectives
            if impact.get('affected_objectives'):
                df = pd.DataFrame(impact['affected_objectives'])
                
                fig = go.Figure()
                
                for _, row in df.iterrows():
                    fig.add_trace(go.Scatter(
                        x=[row['current_score'], row['projected_score']],
                        y=[row['objective_title'][:30], row['objective_title'][:30]],
                        mode='lines+markers',
                        name=row['objective_title'][:30],
                        line=dict(width=3),
                        marker=dict(size=10)
                    ))
                
                fig.update_layout(
                    title="Objective Score Improvements",
                    xaxis_title="Score",
                    xaxis_range=[0, 100],
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)


def render_proposal_card(proposal, index, is_pending=True):
    """Render a proposal card with accept/reject buttons"""
    
    # Priority badge
    if proposal['priority'] == 'high':
        priority_badge = "🔴 HIGH"
        badge_color = "red"
    elif proposal['priority'] == 'medium':
        priority_badge = "🟡 MEDIUM"
        badge_color = "orange"
    else:
        priority_badge = "🟢 LOW"
        badge_color = "green"
    
    with st.container():
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown(f"### {index + 1}. {proposal['action_title']}")
            st.markdown(f"**Objective:** {proposal['objective_title']}")
        
        with col2:
            st.markdown(f"**{priority_badge}**")
        
        # Expandable details
        with st.expander("📋 View Full Details", expanded=False):
            st.write("**Description:**")
            st.write(proposal['description'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Budget Estimate:**")
                st.write(f"${proposal['budget_estimate']:,.0f}")
                
                st.write("**Timeline:**")
                st.write(proposal['timeline'])
            
            with col2:
                st.write("**Expected KPIs:**")
                for kpi in proposal['expected_kpis']:
                    st.write(f"• {kpi}")
            
            st.write("**Rationale:**")
            st.info(proposal['rationale'])
            
            st.write("**Expected Impact:**")
            st.success(proposal['expected_impact'])
        
        # Action buttons (only for pending proposals)
        if is_pending:
            col1, col2, col3 = st.columns([1, 1, 3])
            
            with col1:
                if st.button("✅ Accept", key=f"accept_{proposal['id']}", type="primary", use_container_width=True):
                    with st.spinner("Adding to action plan..."):
                        success = accept_proposal(proposal['id'])
                        if success:
                            st.success("✅ Proposal accepted and added to action plan!")
                            st.info("💡 Re-run the complete analysis to see updated synchronization scores.")
                            st.rerun()
            
            with col2:
                if st.button("❌ Reject", key=f"reject_{proposal['id']}", use_container_width=True):
                    # Mark as rejected (could implement this)
                    st.warning("Proposal rejected")
        else:
            st.success("✅ **This proposal has been accepted and added to the action plan**")
        
        st.markdown("---")


def render_overview(results):
    """Render overview section"""
    st.header("📊 Overall Synchronization Assessment")
    
    score = results['overall_score']
    score_class = "good-score" if score >= 75 else "medium-score" if score >= 60 else "poor-score"
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f'<div class="big-score {score_class}">{score:.1f}/100</div>', unsafe_allow_html=True)
        
        if score >= 90:
            st.success("**Excellent** - Strong alignment")
        elif score >= 75:
            st.info("**Good** - Minor gaps")
        elif score >= 60:
            st.warning("**Moderate** - Improvements needed")
        else:
            st.error("**Poor** - Major misalignment")
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Embedding Score", f"{results['embedding_score']:.1f}/100")
    
    with col2:
        st.metric("Entity Match Score", f"{results['entity_score']:.1f}/100")
    
    with col3:
        strong = results['summary']['objectives_with_strong_support']
        total = results['summary']['total_objectives']
        st.metric("Objectives Supported", f"{strong}/{total}")
    
    with col4:
        matched = results['summary']['matched_entities']
        total_ent = results['summary']['total_strategic_entities']
        match_pct = (matched / total_ent * 100) if total_ent > 0 else 0
        st.metric("Entity Match Rate", f"{match_pct:.0f}%")


def main():
    """Main dashboard function"""
    
    st.title("🎯 Strategic Plan Synchronization Dashboard")
    st.markdown("**AI-Powered Strategic Planning Analysis**")
    
    # Load results
    results = load_results()
    
    if results is None:
        st.error("Results not found. Run analysis first.")
        st.stop()
    
    # Sidebar
    st.sidebar.header("Assessment Details")
    st.sidebar.write(f"**Date:** {results.get('assessment_date', 'N/A')}")
    
    # Check for agent analysis
    agent_results = load_agent_results()
    if agent_results:
        st.sidebar.success(f"🤖 Agent: {agent_results['summary']['total_findings']} findings")
    
    # Navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select View",
        [
            "📊 Overview",
            "💪 Strengths & Weaknesses",
            "💡 Recommendations",
            "🤖 AI Agent Analysis"  # New page
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("AI-powered strategic alignment analysis with autonomous agent capabilities")
    
    # Render selected page
    if page == "📊 Overview":
        render_overview(results)
    elif page == "💪 Strengths & Weaknesses":
        # Implement strengths/weaknesses page
        st.subheader("💪 Strengths")
        for s in results.get('strengths', []):
            st.success(f"• {s}")
        st.subheader("⚠️ Weaknesses")
        for w in results.get('weaknesses', []):
            st.warning(f"• {w}")
    elif page == "💡 Recommendations":
        # Implement recommendations page
        st.subheader("💡 Recommendations")
        for rec in results.get('recommendations', []):
            with st.expander(f"[{rec['priority'].upper()}] {rec.get('objective', 'General')}"):
                for action in rec.get('actions', []):
                    st.write(f"• {action}")
    elif page == "🤖 AI Agent Analysis":
        render_agent_page()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Strategic Planning AI | Powered by OpenAI & Agentic AI"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
