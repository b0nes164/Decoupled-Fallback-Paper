all: paper appendix

paper:
	latexmk paper

appendix:
	latexmk appendix

clean:
	rm -f paper.aux paper.log paper.blg paper.out paper.bbl paper.pdf
	rm -f appendix.aux appendix.log appendix.blg appendix.out appendix.bbl appendix.pdf

