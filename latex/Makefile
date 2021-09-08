LATEXMK_FLAGS ?= ""

texpdf:
	latexmk $(tex_fname) ${LATEXMK_FLAGS} 


report-ch2: tex_fname="report_ch2.tex"
report-ch2: texpdf

report-dev: export LATEXMK_FLAGS="-pvc"
report-dev: report
report: tex_fname="report.tex"
report: texpdf

proposal-dev: export LATEXMK_FLAGS="-pvc"
proposal-dev: proposal
proposal: tex_fname="proposal.tex"
proposal: texpdf

ch1-dev: export LATEXMK_FLAGS="-pvc"
ch1-dev: ch1
ch1: tex_fname="ch1_introduction.tex"
ch1: texpdf

ch2-dev: export LATEXMK_FLAGS="-pvc"
ch2-dev: ch2
ch2: tex_fname="ch2_background.tex"
ch2: texpdf

ch3-dev: export LATEXMK_FLAGS="-pvc"
ch3-dev: ch3
ch3: tex_fname="ch3_methodology.tex"
ch3: texpdf

ch4-dev: export LATEXMK_FLAGS="-pvc"
ch4-dev: ch4
ch4: tex_fname="ch4_experiment.tex"
ch4: texpdf

ch5-dev: export LATEXMK_FLAGS="-pvc"
ch5-dev: ch5
ch5: tex_fname="ch5_conclusion.tex"
ch5: texpdf

mdx-presentation-dev: export LATEXMK_FLAGS="-pvc"
mdx-presentation-dev: mdx-presentation
mdx-presentation: tex_fname="mdx_townhall_aug21.tex"
mdx-presentation: texpdf

all: report

clean:
	latexmk -C
	-rm *.lol
	-rm *.bbl
	-rm *.run.xml
	-rm -r _minted*
	-rm *.nav
	-rm *.pyg
	-rm *.snm
	-rm *.vrb
	-rm *.fls
	-rm *.toc
