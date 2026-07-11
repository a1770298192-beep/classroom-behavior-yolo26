"""Convert the reviewed Markdown draft into the retained GZU LaTeX template."""
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'docs/COURSE_PAPER_DRAFT.md'
OUT=ROOT/'report/main.tex'

FIGS={
'图4-1':('06_system_flow.png','课堂学生行为识别系统总体流程',.98),
'图6-1':('01_overall_metrics.png','不同模型在独立测试集上的检测性能对比',.92),
'图6-2':('02_per_class_ap50.png','不同模型各行为类别AP@0.5对比',.92),
'图6-3':('03_final_vs_baseline_gain.png','最终模型相对基线的逐类别性能变化',.92),
'图6-4':('04_accuracy_speed_tradeoff.png','模型精度—速度权衡',.82),
'图6-5':('05_training_curves.png','三种方案验证集mAP训练曲线',.96),
'图6-6':('07_final_confusion_matrix_normalized.png','最终模型归一化混淆矩阵',.80),
'图7-1':('09_system_demo.jpg','系统单幅图片中文标注效果',.90),
}

def esc(s):
    s=s.replace('—','--').replace('×',r'$\times$').replace('≥',r'$\geq$')
    for a,b in [('\\','@@BS@@'),('&',r'\&'),('%',r'\%'),('#',r'\#'),('_',r'\_'),('$',r'\$')]: s=s.replace(a,b)
    return s.replace('@@BS@@',r'\textbackslash{}')

def inline(s):
    parts=re.split(r'(`[^`]+`|\*\*[^*]+\*\*)',s); out=[]
    for p in parts:
        if p.startswith('`'):
            value=p[1:-1]
            out.append((r'\texttt{\seqsplit{'+value+r'}}') if len(value)>48 and value.isalnum() else (r'\path{'+value+r'}'))
        elif p.startswith('**'): out.append(r'\textbf{'+esc(p[2:-2])+r'}')
        else: out.append(esc(p))
    return ''.join(out)

def parse_table(lines,i):
    rows=[]
    while i<len(lines) and lines[i].startswith('|'):
        cells=[x.strip() for x in lines[i].strip('|').split('|')]
        if not all(re.fullmatch(r':?-+:?',x) for x in cells): rows.append(cells)
        i+=1
    n=len(rows[0]); spec='l'+'c'*(n-1)
    out=[r'\begin{table}[!htbp]',r'\centering',r'\small',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{'+spec+'}',r'\toprule']
    for j,row in enumerate(rows):
        out.append(' & '.join(inline(x) for x in row)+r' \\')
        if j==0: out.append(r'\midrule')
    out += [r'\bottomrule',r'\end{tabular}}',r'\caption{不同实验方案综合性能对比}',r'\label{tab:model-comparison}',r'\end{table}']
    return out,i

def build():
    text=SRC.read_text(encoding='utf-8'); lines=text.splitlines(); body=[]; i=0; in_refs=False
    while i<len(lines):
        s=lines[i].strip()
        if s.startswith('# 基于YOLO26') or s.startswith('> '): i+=1; continue
        if s=='## 摘要': body += [r'\pagenumbering{Roman}',r'\begin{abstract}']; i+=1; continue
        if s=='## Abstract': body += [r'\end{abstract}',r'\begin{center}\textbf{Abstract}\end{center}']; i+=1; continue
        if s.startswith('**关键词：**'):
            body.append(r'\par\noindent\textbf{关键词：}'+inline(s.split('**',2)[2].lstrip())); i+=1; continue
        if s.startswith('**Key words:**'):
            body.append(r'\par\noindent\textbf{Key words:}'+inline(s.split('**',2)[2].lstrip())); i+=1; continue
        if s=='# 1 绪论':
            body += [r'\newpage',r'\tableofcontents',r'\newpage',r'\pagenumbering{arabic}',r'\setcounter{page}{1}',r'\section{绪论}']; i+=1; continue
        if s=='# 参考文献': in_refs=True; body += [r'\clearpage',r'\begin{thebibliography}{99}']; i+=1; continue
        if in_refs and re.match(r'^\[\d+\]',s):
            m=re.match(r'^\[(\d+)\]\s*(.*)',s); body.append(r'\bibitem{ref'+m.group(1)+'} '+inline(m.group(2))); i+=1; continue
        if in_refs and s.startswith('## 附录'):
            body += [r'\end{thebibliography}',r'\appendix',r'\section{'+inline(s.split(' ',2)[2])+r'}']; in_refs=False; i+=1; continue
        if s.startswith('## 附录'):
            body += [r'\section{'+inline(s.split(' ',2)[2])+r'}']; i+=1; continue
        if s.startswith('# '): body.append(r'\section{'+inline(re.sub(r'^\d+\s*','',s[2:]))+r'}'); i+=1; continue
        if s.startswith('## '): body.append(r'\subsection{'+inline(re.sub(r'^\d+\.\d+\s*','',s[3:]))+r'}'); i+=1; continue
        if s.startswith('### '): body.append(r'\subsubsection{'+inline(s[4:])+r'}'); i+=1; continue
        if s.startswith('【插图位置：'):
            key=next((k for k in FIGS if k in s),None)
            if key:
                fn,cap,w=FIGS[key]; body += [r'\begin{figure}[!htbp]',r'\centering',rf'\includegraphics[width={w}\textwidth]{{figures/{fn}}}',r'\caption{'+cap+r'}',r'\end{figure}']
            i+=1; continue
        if s.startswith('|'):
            out,i=parse_table(lines,i); body+=out; continue
        if re.match(r'^P = TP',s):
            body += [r'\begin{equation}',r'P=\frac{TP}{TP+FP},\qquad R=\frac{TP}{TP+FN}.',r'\end{equation}']; i+=1; continue
        if s: body.append(inline(s)+r'\par')
        i+=1
    if in_refs: body.append(r'\end{thebibliography}')
    pre=r'''%!TeX program = xelatex
\documentclass[12pt,hyperref,a4paper,UTF8]{ctexart}
\usepackage{UCASReport}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{caption}
\usepackage{setspace}
\usepackage{float}
\usepackage{seqsplit}
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}
\setlength{\parindent}{2em}
\setlength{\headheight}{15pt}
\setlength{\emergencystretch}{3em}
\setlength{\parskip}{0pt}
\linespread{1.45}
\captionsetup{font=small,labelsep=quad}
\begin{document}
\cover
\thispagestyle{empty}
\clearpage
'''
    OUT.write_text(pre+'\n'.join(body)+'\n\\end{document}\n',encoding='utf-8')
    print(OUT)

if __name__=='__main__': build()
