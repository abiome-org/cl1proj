TEX := pdflatex
BUILD_DIR := build
MAIN := paper

.PHONY: all clean distclean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex figures/*.pdf
	mkdir -p $(BUILD_DIR)
	$(TEX) -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=$(BUILD_DIR) $(MAIN).tex
	$(TEX) -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=$(BUILD_DIR) $(MAIN).tex
	cp $(BUILD_DIR)/$(MAIN).pdf $@

clean:
	rm -rf $(BUILD_DIR)

distclean: clean
	rm -f $(MAIN).pdf
