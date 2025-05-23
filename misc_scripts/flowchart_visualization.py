from graphviz import Digraph

dot = Digraph()
# imaging nodes
dot.node('A', label=r"""<
    <TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0">
        <TR><TD><B>Patient Class</B></TD></TR>
        <TR><TD>Stores metadata</TD></TR>
        <TR><TD>Represents a patient with N metastases over time</TD></TR>
    </TABLE>
>""")

dot.node('B', label=r"""<
    <TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0">
        <TR><TD><B>Time Series Class</B></TD></TR>
        <TR><TD>Stores Dates and Metastases</TD></TR>
        <TR><TD>Represents multiple metastases with temporal attachment</TD></TR>
    </TABLE>
>""")

dot.node('C', label=r"""<
    <TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0">
        <TR><TD><B>Metastasis Class</B></TD></TR>
        <TR><TD>Stores Lesion Volume</TD></TR>
        <TR><TD>Represents a single metastasis, no temporal attachment</TD></TR>
    </TABLE>
>""")
dot.node('F', 'Empty Metastasis Class')
dot.node('D', 'Interpolated Metastasis Class')

# dot.node('E', 'Relevant Images')
# dot.node('G', 'Segmentations & Dose')
# dot.node('H', 'Filter/Register')
# dot.node('I', 'nnUNet Resegmentation')
# dot.node('J', 'Preprocessed Dataset')

# dot.edges(['CD', 'BD', 'DE', 'AF', 'FG', 'HI', 'IJ'])
dot.edge('C', 'F', label='Inherits')
dot.edge('C', 'D', label='Inherits')
dot.edge('C', 'B', label='Form')
dot.edge('F', 'B', label='Form')
dot.edge('D', 'B', label='Form')
dot.edge('B', 'A', label='Form')
dot.edge('A', 'B', label='Parses')


dot.render('analysis_flowchart', format='png')