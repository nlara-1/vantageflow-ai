"""
Professional PDF underwriting report generator for VantageFlow AI.

Creates bank-style credit decision reports with score display, reason codes,
financial metrics, and embedded visualizations using ReportLab.
"""

import os
from typing import List, Dict, Optional, Any
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether
)
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


class UnderwritingReportGenerator:
    """
    Generate professional PDF underwriting reports.

    Creates comprehensive credit decision reports with:
    - Borrower information and decision summary
    - Prominent credit score display
    - Positive and negative contributing factors
    - Financial metrics table
    - Embedded visualizations
    - Professional bank-style formatting
    """

    def __init__(self, company_name: str = "VantageFlow AI"):
        """
        Initialize report generator.

        Args:
            company_name: Name of the organization (appears in header)
        """
        self.company_name = company_name
        self.page_width, self.page_height = letter

        # Color scheme (professional bank colors)
        self.colors = {
            'primary': colors.HexColor('#1E3A8A'),      # Dark blue
            'secondary': colors.HexColor('#3B82F6'),    # Medium blue
            'success': colors.HexColor('#059669'),      # Green
            'warning': colors.HexColor('#D97706'),      # Orange
            'danger': colors.HexColor('#DC2626'),       # Red
            'neutral': colors.HexColor('#6B7280'),      # Gray
            'light_bg': colors.HexColor('#F3F4F6'),     # Light gray background
            'score_box': colors.HexColor('#EFF6FF')     # Light blue background
        }

        # Initialize styles
        self._setup_styles()

    def _setup_styles(self) -> None:
        """Setup custom paragraph styles for the report."""
        self.styles = getSampleStyleSheet()

        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=self.colors['primary'],
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=self.colors['primary'],
            spaceAfter=6,
            spaceBefore=12,
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderColor=self.colors['primary'],
            borderPadding=4,
            backColor=self.colors['light_bg']
        ))

        # Score display style
        self.styles.add(ParagraphStyle(
            name='ScoreDisplay',
            parent=self.styles['Normal'],
            fontSize=48,
            textColor=self.colors['primary'],
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        # Score label style
        self.styles.add(ParagraphStyle(
            name='ScoreLabel',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=self.colors['neutral'],
            alignment=TA_CENTER,
            fontName='Helvetica'
        ))

        # Reason code style
        self.styles.add(ParagraphStyle(
            name='ReasonCode',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.black,
            spaceAfter=4,
            fontName='Helvetica'
        ))

        # Body text style
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.black,
            spaceAfter=6,
            fontName='Helvetica'
        ))

    def generate_report(
        self,
        borrower_id: str,
        score: float,
        risk_tier: str,
        reason_codes: List[Dict[str, Any]],
        financial_metrics: Dict[str, float],
        charts: Optional[Dict[str, str]] = None,
        output_path: str = "underwriting_report.pdf",
        borrower_name: Optional[str] = None,
        decision: str = "PENDING"
    ) -> str:
        """
        Generate complete underwriting report.

        Args:
            borrower_id: Unique borrower identifier
            score: Credit score (0-100 or 0-1000 scale)
            risk_tier: Risk category ('LOW', 'MEDIUM', 'HIGH', 'VERY HIGH')
            reason_codes: List of reason code dicts with keys:
                - code: Reason code ID (e.g., 'N01')
                - name: Reason name
                - description: Short description
                - long_description: Detailed explanation
                - impact: 'positive' or 'negative'
                - magnitude: 'strong', 'moderate', or 'slight'
                - contribution: SHAP value
            financial_metrics: Dictionary of metrics to display
            charts: Optional dict mapping chart names to file paths
            output_path: Output PDF file path
            borrower_name: Optional borrower name
            decision: Credit decision ('APPROVED', 'DENIED', 'PENDING', 'MANUAL_REVIEW')

        Returns:
            Path to generated PDF file

        Example:
            >>> from src.reporting.generator import UnderwritingReportGenerator
            >>>
            >>> generator = UnderwritingReportGenerator(company_name="VantageFlow AI")
            >>>
            >>> reason_codes = [
            ...     {
            ...         'code': 'N01',
            ...         'name': 'INCOME_VOLATILITY_HIGH',
            ...         'description': 'Income volatility is higher than typical',
            ...         'long_description': 'Your income shows significant month-to-month variation...',
            ...         'impact': 'negative',
            ...         'magnitude': 'strong',
            ...         'contribution': 0.045
            ...     },
            ...     {
            ...         'code': 'P01',
            ...         'name': 'SAVINGS_BEHAVIOR_POSITIVE',
            ...         'description': 'Strong savings behavior with positive cash flow',
            ...         'long_description': 'Your account shows consistent savings...',
            ...         'impact': 'positive',
            ...         'magnitude': 'moderate',
            ...         'contribution': 0.028
            ...     }
            ... ]
            >>>
            >>> metrics = {
            ...     'avg_monthly_income': 4200.50,
            ...     'savings_rate': 0.15,
            ...     'expense_income_ratio': 0.82,
            ...     'overdraft_count_3mo': 2,
            ...     'income_cv': 0.35
            ... }
            >>>
            >>> charts = {
            ...     'waterfall': 'output/charts/shap_waterfall.png',
            ...     'score_distribution': 'output/charts/score_dist.png'
            ... }
            >>>
            >>> pdf_path = generator.generate_report(
            ...     borrower_id="12345",
            ...     score=680,
            ...     risk_tier="MEDIUM",
            ...     reason_codes=reason_codes,
            ...     financial_metrics=metrics,
            ...     charts=charts,
            ...     output_path="output/reports/underwriting_report.pdf",
            ...     borrower_name="John Doe",
            ...     decision="APPROVED_WITH_CONDITIONS"
            ... )
            >>> print(f"Report generated: {pdf_path}")
        """
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        # Initialize PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
            leftMargin=0.75*inch,
            rightMargin=0.75*inch
        )

        # Build report elements
        elements = []

        # 1. Header section
        elements.extend(self._build_header(borrower_id, borrower_name, decision))
        elements.append(Spacer(1, 0.3*inch))

        # 2. Score display box (prominent)
        elements.extend(self._build_score_box(score, risk_tier))
        elements.append(Spacer(1, 0.4*inch))

        # 3. Positive factors
        positive_codes = [rc for rc in reason_codes if rc.get('impact') == 'positive']
        if positive_codes:
            elements.extend(self._build_reason_section(
                "Positive Contributing Factors",
                positive_codes[:3],
                is_positive=True
            ))
            elements.append(Spacer(1, 0.3*inch))

        # 4. Negative factors
        negative_codes = [rc for rc in reason_codes if rc.get('impact') == 'negative']
        if negative_codes:
            elements.extend(self._build_reason_section(
                "Factors Requiring Attention",
                negative_codes[:3],
                is_positive=False
            ))
            elements.append(Spacer(1, 0.3*inch))

        # 5. Financial metrics table
        elements.extend(self._build_metrics_table(financial_metrics))
        elements.append(Spacer(1, 0.3*inch))

        # 6. Embedded charts
        if charts:
            elements.extend(self._build_charts_section(charts))

        # 7. Footer/disclaimer
        elements.extend(self._build_footer())

        # Build PDF
        doc.build(elements, onFirstPage=self._add_page_number, onLaterPages=self._add_page_number)

        print(f"✓ Underwriting report generated: {output_path}")
        return output_path

    def _build_header(
        self,
        borrower_id: str,
        borrower_name: Optional[str],
        decision: str
    ) -> List[Any]:
        """Build report header section."""
        elements = []

        # Company name and title
        elements.append(Paragraph(
            self.company_name,
            self.styles['ReportTitle']
        ))
        elements.append(Paragraph(
            "Credit Underwriting Report",
            ParagraphStyle(
                'Subtitle',
                parent=self.styles['Normal'],
                fontSize=14,
                textColor=self.colors['neutral'],
                alignment=TA_CENTER,
                spaceAfter=12
            )
        ))

        # Report metadata table
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M")

        metadata = [
            ['Report Date:', report_date, 'Borrower ID:', borrower_id],
        ]

        if borrower_name:
            metadata.append(['Borrower Name:', borrower_name, 'Decision:', decision])
        else:
            metadata.append(['Decision:', decision, '', ''])

        metadata_table = Table(metadata, colWidths=[1.5*inch, 2*inch, 1.5*inch, 2*inch])
        metadata_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 9),
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 9),
            ('FONT', (2, 0), (2, -1), 'Helvetica-Bold', 9),
            ('TEXTCOLOR', (0, 0), (-1, -1), self.colors['neutral']),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))

        elements.append(metadata_table)

        return elements

    def _build_score_box(self, score: float, risk_tier: str) -> List[Any]:
        """Build prominent score display box."""
        elements = []

        # Determine score color based on risk tier
        risk_colors = {
            'LOW': self.colors['success'],
            'MEDIUM': self.colors['warning'],
            'HIGH': self.colors['danger'],
            'VERY HIGH': self.colors['danger']
        }
        score_color = risk_colors.get(risk_tier, self.colors['neutral'])

        # Create score box content
        score_data = [
            [Paragraph(
                f"<b>{int(score)}</b>",
                ParagraphStyle(
                    'ScoreLarge',
                    parent=self.styles['ScoreDisplay'],
                    textColor=score_color
                )
            )],
            [Paragraph("Credit Score", self.styles['ScoreLabel'])],
            [Paragraph(
                f"<b>Risk Tier: {risk_tier}</b>",
                ParagraphStyle(
                    'RiskTier',
                    parent=self.styles['ScoreLabel'],
                    fontSize=14,
                    textColor=score_color
                )
            )]
        ]

        score_table = Table(score_data, colWidths=[6*inch])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), self.colors['score_box']),
            ('BOX', (0, 0), (-1, -1), 2, self.colors['primary']),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))

        elements.append(score_table)

        return elements

    def _build_reason_section(
        self,
        title: str,
        reason_codes: List[Dict[str, Any]],
        is_positive: bool
    ) -> List[Any]:
        """Build reason codes section (positive or negative factors)."""
        elements = []

        # Section header
        elements.append(Paragraph(title, self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1*inch))

        # Reason codes
        for i, rc in enumerate(reason_codes, 1):
            code = rc.get('code', 'N/A')
            name = rc.get('name', 'Unknown')
            description = rc.get('description', '')
            long_description = rc.get('long_description', '')
            magnitude = rc.get('magnitude', 'unknown').upper()
            contribution = rc.get('contribution', 0.0)

            # Color based on impact
            impact_color = self.colors['success'] if is_positive else self.colors['danger']
            impact_symbol = "[+]" if is_positive else "[-]"

            # Build reason code display
            reason_data = [
                [
                    Paragraph(
                        f"<b>{i}. {impact_symbol} {name}</b>",
                        ParagraphStyle(
                            'ReasonTitle',
                            parent=self.styles['ReasonCode'],
                            fontSize=11,
                            textColor=impact_color,
                            fontName='Helvetica-Bold'
                        )
                    ),
                    Paragraph(
                        f"<b>Code:</b> {code}<br/><b>Magnitude:</b> {magnitude}",
                        ParagraphStyle(
                            'ReasonMeta',
                            parent=self.styles['ReasonCode'],
                            fontSize=9,
                            textColor=self.colors['neutral'],
                            alignment=TA_RIGHT
                        )
                    )
                ],
                [
                    Paragraph(description, self.styles['BodyText']),
                    ''
                ],
                [
                    Paragraph(
                        f"<i>{long_description}</i>",
                        ParagraphStyle(
                            'ReasonDetail',
                            parent=self.styles['BodyText'],
                            fontSize=9,
                            textColor=self.colors['neutral'],
                            leftIndent=10
                        )
                    ),
                    ''
                ]
            ]

            reason_table = Table(reason_data, colWidths=[4.5*inch, 1.5*inch])
            reason_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.white),
                ('BOX', (0, 0), (-1, -1), 0.5, self.colors['neutral']),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ]))

            elements.append(reason_table)
            elements.append(Spacer(1, 0.15*inch))

        return elements

    def _build_metrics_table(self, metrics: Dict[str, float]) -> List[Any]:
        """Build financial metrics table."""
        elements = []

        # Section header
        elements.append(Paragraph("Financial Metrics Summary", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1*inch))

        # Format metrics
        formatted_metrics = self._format_metrics(metrics)

        # Build table data (2 columns)
        table_data = [['Metric', 'Value', 'Metric', 'Value']]

        metrics_list = list(formatted_metrics.items())
        for i in range(0, len(metrics_list), 2):
            row = []

            # First metric
            metric1_name, metric1_value = metrics_list[i]
            row.extend([metric1_name, metric1_value])

            # Second metric (if exists)
            if i + 1 < len(metrics_list):
                metric2_name, metric2_value = metrics_list[i + 1]
                row.extend([metric2_name, metric2_value])
            else:
                row.extend(['', ''])

            table_data.append(row)

        metrics_table = Table(table_data, colWidths=[2*inch, 1*inch, 2*inch, 1*inch])
        metrics_table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['primary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),

            # Data rows
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('FONT', (0, 1), (0, -1), 'Helvetica', 9),
            ('FONT', (1, 1), (1, -1), 'Helvetica-Bold', 9),
            ('FONT', (2, 1), (2, -1), 'Helvetica', 9),
            ('FONT', (3, 1), (3, -1), 'Helvetica-Bold', 9),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ('ALIGN', (3, 1), (3, -1), 'RIGHT'),

            # Borders
            ('BOX', (0, 0), (-1, -1), 1, self.colors['neutral']),
            ('LINEBELOW', (0, 0), (-1, 0), 1, colors.white),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, self.colors['light_bg']),

            # Padding
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))

        elements.append(metrics_table)

        return elements

    def _format_metrics(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Format metrics for display."""
        formatted = {}

        for key, value in metrics.items():
            # Convert snake_case to Title Case
            display_name = key.replace('_', ' ').title()

            # Format value based on metric type
            if 'rate' in key or 'ratio' in key or 'pct' in key or key.endswith('_cv'):
                # Percentage metrics
                formatted[display_name] = f"{value * 100:.1f}%"
            elif 'income' in key or 'balance' in key or 'spending' in key or 'cashflow' in key:
                # Currency metrics
                formatted[display_name] = f"${value:,.2f}"
            elif 'count' in key or 'frequency' in key:
                # Count metrics
                formatted[display_name] = f"{int(value)}"
            else:
                # Generic numeric
                formatted[display_name] = f"{value:.3f}"

        return formatted

    def _build_charts_section(self, charts: Dict[str, str]) -> List[Any]:
        """Build embedded charts section."""
        elements = []

        # Section header
        elements.append(Paragraph("Visual Analysis", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1*inch))

        # Add each chart
        for chart_name, chart_path in charts.items():
            if not os.path.exists(chart_path):
                print(f"⚠ Warning: Chart not found: {chart_path}")
                continue

            # Chart title
            chart_title = chart_name.replace('_', ' ').title()
            elements.append(Paragraph(
                f"<b>{chart_title}</b>",
                ParagraphStyle(
                    'ChartTitle',
                    parent=self.styles['BodyText'],
                    fontSize=11,
                    spaceAfter=6
                )
            ))

            # Add image (constrained to fit page width)
            try:
                img = Image(chart_path, width=6*inch, height=3.5*inch, kind='proportional')
                elements.append(img)
                elements.append(Spacer(1, 0.2*inch))
            except Exception as e:
                print(f"⚠ Warning: Could not embed chart {chart_path}: {str(e)}")

        return elements

    def _build_footer(self) -> List[Any]:
        """Build report footer with disclaimer."""
        elements = []

        elements.append(Spacer(1, 0.2*inch))

        disclaimer = """
        <b>IMPORTANT DISCLAIMER:</b><br/>
        This credit assessment is based on alternative data analysis and behavioral patterns.
        The decision is made in compliance with the Fair Credit Reporting Act (FCRA) Section 615(a)
        and Equal Credit Opportunity Act (ECOA). If you have questions about this decision or
        believe any information is inaccurate, please contact us immediately. You have the right
        to request additional information about the factors that affected this decision.
        """

        elements.append(Paragraph(
            disclaimer,
            ParagraphStyle(
                'Disclaimer',
                parent=self.styles['BodyText'],
                fontSize=8,
                textColor=self.colors['neutral'],
                borderWidth=1,
                borderColor=self.colors['neutral'],
                borderPadding=8,
                backColor=self.colors['light_bg']
            )
        ))

        return elements

    def _add_page_number(self, canvas_obj: canvas.Canvas, doc: Any) -> None:
        """Add page numbers to each page."""
        page_num = canvas_obj.getPageNumber()
        text = f"Page {page_num}"
        canvas_obj.setFont('Helvetica', 9)
        canvas_obj.setFillColor(self.colors['neutral'])
        canvas_obj.drawRightString(
            self.page_width - 0.75*inch,
            0.5*inch,
            text
        )


if __name__ == "__main__":
    """
    Example usage: Generate sample underwriting report.
    """
    print("Underwriting Report Generator")
    print("\nExample usage:")
    print("  from src.reporting.generator import UnderwritingReportGenerator")
    print("  generator = UnderwritingReportGenerator()")
    print("  generator.generate_report(borrower_id, score, risk_tier, reason_codes, metrics)")
    print("\nSee docstring for complete example with all parameters.")
