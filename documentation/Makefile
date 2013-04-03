#
# Makefile for pdfs
#

TEXINPUTS:=.//:$(TEXINPUTS)
PDFLATEX=pdflatex -shell-escape
# use bibtex
BIBLATEX = bibtex

ifeq ($(shell echo -e x), -e x)
    ECHO = echo
else
    ECHO = echo -e
endif

TEX2PAGE = my-tex2page

TEX = documentation.tex
BIB = literature.bib
PDF = $(TEX:%.tex=%.pdf)
XHTML = $(TEX:%.tex=hipacc_%.html)

default: $(PDF)

pdf: $(PDF)
html: $(XHTML)
all: pdf html


$(PDF): *.tex $(BIB)
	@$(ECHO) " *\n * pdflatex: $(TEX) > $@ \n *"; \
	( \
	TEXINPUTS=$(TEXINPUTS) $(PDFLATEX) $(TEX); \
	$(BIBLATEX) $(TEX:.tex=); \
	TEXINPUTS=$(TEXINPUTS) $(PDFLATEX) $(TEX); \
	if grep -q "There were undefined references." $(TEX:.tex=.log); \
	then \
		TEXINPUTS=$(TEXINPUTS) $(PDFLATEX) $(TEX); \
	fi; \
	while grep -q "Rerun to get cross-references right." $(TEX:.tex=.log); \
	do \
		TEXINPUTS=$(TEXINPUTS) $(PDFLATEX) $(TEX); \
	done; \
	$(ECHO) "\n\n *******************************************************************************"; \
	$(ECHO) " *                                                                             *"; \
	$(ECHO) " *   WARNING SUMMARY                                                           *"; \
	$(ECHO) " *                                                                             *"; \
	grep -i "Warning" $(TEX:.tex=.log); \
	$(ECHO) " *                                                                             *"; \
	$(ECHO) " *******************************************************************************\n"; \
	)

XTEX = hipacc_documentation.tex
ALL_TEX = $(shell ls *.tex)
$(XHTML): *.tex $(BIB)
	$(TEX2PAGE) $(XTEX)
	$(TEX2PAGE) $(XTEX)
	sed -e 's~</head~<link href="prettify/prettify.css" type="text/css" rel="stylesheet" /><script type="text/javascript" src="prettify/prettify.js"></script></head~g' -e 's~<body~<body onload="prettyPrint()" ~g' $(XHTML) > $(XHTML:.html=.html_pre)
	mv $(XHTML:.html=.html_pre) $(XHTML)



.PHONY: all clean default distclean html pdf
clean:
	@$(ECHO) " ** Remove automatically generated files "; \
	rm -f *.out *.bbl *.blg *.log *.toc *.idx *.aux; \
	rm -f *.dep *.dpth *.auxlock *.sub *.suc *.syc *.sym; \
	rm -f $(ALL_TEX:%.tex=%-Z*) $(ALL_TEX:%.tex=%.hlog);


distclean: clean
	@rm -f $(PDF) $(XHTML);

