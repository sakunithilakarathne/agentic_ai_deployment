"""
Document Processor for Strategic and Action Plans
Extracts text from PDFs and structures the data into JSON format
"""

import re
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import PyPDF2
from pathlib import Path


@dataclass
class KPI:
    """Represents a Key Performance Indicator"""
    metric: str
    target: Optional[float] = None
    unit: Optional[str] = None
    baseline: Optional[float] = None
    deadline: Optional[str] = None


@dataclass
class Section:
    """Represents a section in the document (objective or action)"""
    id: str
    type: str  # "strategic_objective" or "action_item"
    title: str
    content: str
    kpis: List[KPI]
    budget: Optional[float] = None
    timeline: Optional[str] = None
    initiatives: List[str] = None
    priority: Optional[str] = None
    
    def __post_init__(self):
        if self.initiatives is None:
            self.initiatives = []


@dataclass
class Document:
    """Represents the complete document"""
    document_type: str  # "strategic_plan" or "action_plan"
    title: str
    organization: str
    planning_period: str
    sections: List[Section]
    total_budget: Optional[float] = None


class DocumentProcessor:
    """Processes Strategic and Action Plans from PDF format"""
    
    def __init__(self):
        self.strategic_patterns = {
            'objective': r'(?:STRATEGIC OBJECTIVE|### STRATEGIC OBJECTIVE)\s+(\d+):\s*([^\n]+)',
            'section_header': r'(?:###|##)\s+([A-Z][^\n]+)',
        }
        
        self.action_patterns = {
            'action': r'(?:ACTION|### ACTION)\s+([\d\.]+):\s*([^\n]+)',
            'initiative': r'Initiative Lead:|Budget:|Timeline:|Priority:',
        }
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            if str(file_path).endswith('.md'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif str(file_path).endswith('.pdf'):
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
        
        return text
    
    def extract_metadata(self, text: str) -> Dict[str, str]:
        """Extract document metadata"""
        metadata = {
            'title': '',
            'organization': '',
            'planning_period': ''
        }
        
        # Extract title (first # heading)
        title_match = re.search(r'^#\s+(.+?)(?:\n|$)', text, re.MULTILINE)
        if title_match:
            metadata['title'] = title_match.group(1).strip()
        
        # Extract organization (second ## heading)
        org_match = re.search(r'^##\s+(.+?)(?:\n|$)', text, re.MULTILINE)
        if org_match:
            metadata['organization'] = org_match.group(1).strip()
        
        # Extract planning period
        period_match = re.search(r'(?:Planning Period|Period):\s*([^\n]+)', text, re.IGNORECASE)
        if period_match:
            metadata['planning_period'] = period_match.group(1).strip()
        else:
            # Try to find year range
            year_match = re.search(r'\b(20\d{2})\s*[-–—]\s*(20\d{2})\b', text)
            if year_match:
                metadata['planning_period'] = f"{year_match.group(1)}-{year_match.group(2)}"
        
        return metadata
    
    def extract_kpis(self, text: str) -> List[KPI]:
        """Extract KPIs from text"""
        kpis = []
        
        # Pattern 1: "metric from X to Y by deadline"
        pattern1 = r'([A-Za-z][A-Za-z\s-]+?)\s+from\s+([\d.]+)(%|M|B|ratio)?\s+to\s+([\d.]+)(%|M|B|ratio)?\s+(?:by|in)\s+(Q\d\s+\d{4}|\d{4})'
        matches1 = re.finditer(pattern1, text, re.IGNORECASE)
        for match in matches1:
            metric = match.group(1).strip()
            baseline = float(match.group(2))
            target = float(match.group(4))
            unit = match.group(5) or match.group(3) or ""
            deadline = match.group(6)
            
            kpis.append(KPI(
                metric=metric,
                baseline=baseline,
                target=target,
                unit=unit,
                deadline=deadline
            ))
        
        # Pattern 2: "metric: target value by deadline"
        pattern2 = r'([A-Za-z][A-Za-z\s-]+?):\s*([\d.]+)(%|M|B|ratio)?\s+(?:by|in)\s+(Q\d\s+\d{4}|\d{4})'
        matches2 = re.finditer(pattern2, text, re.IGNORECASE)
        for match in matches2:
            metric = match.group(1).strip()
            target = float(match.group(2))
            unit = match.group(3) or ""
            deadline = match.group(4)
            
            if not any(kpi.metric.lower() == metric.lower() for kpi in kpis):
                kpis.append(KPI(
                    metric=metric,
                    target=target,
                    unit=unit,
                    deadline=deadline
                ))
        
        # Pattern 3: "target: value" in KPI sections
        pattern3 = r'(?:target|goal):\s*([\d.]+)(%|M|B|\+|ratio)?'
        in_kpi_section = 'KPI' in text or 'Key Performance' in text
        if in_kpi_section:
            matches3 = re.finditer(pattern3, text, re.IGNORECASE)
            for match in matches3:
                target = float(match.group(1))
                unit = match.group(2) or ""
                
                # Try to find metric name before the target
                context_start = max(0, match.start() - 100)
                context = text[context_start:match.start()]
                metric_match = re.search(r'([A-Z][A-Za-z\s-]+?)(?:\s*:|\s*\()', context)
                
                if metric_match:
                    metric = metric_match.group(1).strip()
                    if not any(kpi.metric.lower() == metric.lower() for kpi in kpis):
                        kpis.append(KPI(
                            metric=metric,
                            target=target,
                            unit=unit
                        ))
        
        return kpis
    
    def extract_budget(self, text: str) -> Optional[float]:
        """Extract budget amount from text"""
        # Pattern: $XX.XM or $XXM or $X.XB
        pattern = r'\$\s*([\d,]+(?:\.\d+)?)\s*(M|million|B|billion)?'
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        budgets = []
        for match in matches:
            amount_str = match.group(1).replace(',', '')
            amount = float(amount_str)
            unit = match.group(2)
            
            if unit:
                unit_lower = unit.lower()
                if unit_lower in ['m', 'million']:
                    amount *= 1_000_000
                elif unit_lower in ['b', 'billion']:
                    amount *= 1_000_000_000
            
            budgets.append(amount)
        
        # Return the largest budget found (most likely the main budget)
        return max(budgets) if budgets else None
    
    def extract_timeline(self, text: str) -> Optional[str]:
        """Extract timeline from text"""
        # Pattern 1: Q1 2025 - Q4 2026
        pattern1 = r'(Q\d\s+\d{4})\s*[-–—]\s*(Q\d\s+\d{4})'
        match1 = re.search(pattern1, text)
        if match1:
            return f"{match1.group(1)} - {match1.group(2)}"
        
        # Pattern 2: 2025-2028
        pattern2 = r'(20\d{2})\s*[-–—]\s*(20\d{2})'
        match2 = re.search(pattern2, text)
        if match2:
            return f"{match2.group(1)}-{match2.group(2)}"
        
        # Pattern 3: Timeline: Q1 2025
        pattern3 = r'Timeline:\s*([^\n]+)'
        match3 = re.search(pattern3, text, re.IGNORECASE)
        if match3:
            return match3.group(1).strip()
        
        return None
    
    def extract_initiatives(self, text: str) -> List[str]:
        """Extract initiative names from text"""
        initiatives = []
        
        # Look for numbered or bulleted lists
        patterns = [
            r'^\s*[\d]+\.\s*\*\*([^*]+)\*\*',  # 1. **Initiative Name**
            r'^\s*-\s*\*\*([^*]+)\*\*',         # - **Initiative Name**
            r'^\s*[\d]+\.\s*([A-Z][^\n:]+?)(?:\n|:)',  # 1. Initiative Name
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                initiative = match.group(1).strip()
                if len(initiative) > 5 and len(initiative) < 100:  # Reasonable length
                    initiatives.append(initiative)
        
        return initiatives
    
    def extract_priority(self, text: str) -> Optional[str]:
        """Extract priority level"""
        pattern = r'Priority:\s*(Critical|High|Medium|Low)'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).capitalize() if match else None
    
    def split_into_sections(self, text: str, doc_type: str) -> List[Dict[str, Any]]:
        """Split document into major sections"""
        sections = []
        
        if doc_type == "strategic_plan":
            # Find strategic objectives
            pattern = r'### STRATEGIC OBJECTIVE (\d+): ([^\n]+)'
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            for i, match in enumerate(matches):
                section_id = f"obj_{match.group(1)}"
                title = match.group(2).strip()
                
                # Extract content until next objective or end
                start_pos = match.end()
                if i < len(matches) - 1:
                    end_pos = matches[i + 1].start()
                else:
                    end_pos = len(text)
                
                content = text[start_pos:end_pos].strip()
                
                sections.append({
                    'id': section_id,
                    'type': 'strategic_objective',
                    'title': title,
                    'content': content
                })
        
        elif doc_type == "action_plan":
            # Find action items
            pattern = r'### ACTION ([\d\.]+): ([^\n]+)'
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            for i, match in enumerate(matches):
                section_id = f"action_{match.group(1).replace('.', '_')}"
                title = match.group(2).strip()
                
                # Extract content until next action or end
                start_pos = match.end()
                if i < len(matches) - 1:
                    end_pos = matches[i + 1].start()
                else:
                    end_pos = len(text)
                
                content = text[start_pos:end_pos].strip()
                
                sections.append({
                    'id': section_id,
                    'type': 'action_item',
                    'title': title,
                    'content': content
                })
        
        return sections
    
    def process_document(self, pdf_path: str, doc_type: str) -> Document:
        """
        Main processing function
        
        Args:
            pdf_path: Path to PDF file
            doc_type: Either 'strategic_plan' or 'action_plan'
        
        Returns:
            Document object with structured data
        """
        # Extract text from PDF
        print(f"Extracting text from {pdf_path}...")
        text = self.extract_text_from_pdf(pdf_path)
        
        # Extract metadata
        print("Extracting metadata...")
        metadata = self.extract_metadata(text)
        
        # Split into sections
        print(f"Parsing sections for {doc_type}...")
        raw_sections = self.split_into_sections(text, doc_type)
        
        # Process each section
        processed_sections = []
        total_budget = 0
        
        for raw_section in raw_sections:
            print(f"  Processing: {raw_section['title'][:50]}...")
            
            section = Section(
                id=raw_section['id'],
                type=raw_section['type'],
                title=raw_section['title'],
                content=raw_section['content'],
                kpis=self.extract_kpis(raw_section['content']),
                budget=self.extract_budget(raw_section['content']),
                timeline=self.extract_timeline(raw_section['content']),
                initiatives=self.extract_initiatives(raw_section['content']),
                priority=self.extract_priority(raw_section['content'])
            )
            
            if section.budget:
                total_budget += section.budget
            
            processed_sections.append(section)
        
        # Create document
        document = Document(
            document_type=doc_type,
            title=metadata['title'],
            organization=metadata['organization'],
            planning_period=metadata['planning_period'],
            sections=processed_sections,
            total_budget=total_budget if total_budget > 0 else None
        )
        
        print(f"✓ Extracted {len(processed_sections)} sections")
        print(f"✓ Total KPIs: {sum(len(s.kpis) for s in processed_sections)}")
        print(f"✓ Total Budget: ${total_budget:,.0f}" if total_budget > 0 else "✓ No budget found")
        
        return document
    
    def to_json(self, document: Document, output_path: Optional[str] = None) -> str:
        """Convert document to JSON format"""
        # Convert dataclasses to dict
        doc_dict = asdict(document)
        
        # Convert to JSON string
        json_str = json.dumps(doc_dict, indent=2)
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            print(f"✓ Saved JSON to {output_path}")
        
        return json_str


# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    # Process Strategic Plan
    strategic_doc = processor.process_document(
        pdf_path="strategic_plan.pdf",
        doc_type="strategic_plan"
    )
    
    # Process Action Plan
    action_doc = processor.process_document(
        pdf_path="action_plan.pdf",
        doc_type="action_plan"
    )
    
    # Save to JSON
    processor.to_json(strategic_doc, "strategic_plan.json")
    processor.to_json(action_doc, "action_plan.json")
    
    print("\n✓ Document processing complete!")
