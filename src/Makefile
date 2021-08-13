BIN  = ../bin

.PHONY: install clean distclean

install:
	. /etc/environ.sh; use -e -r anaconda-7; python -m compileall API.py
	mv __pycache__/API.cpython-38.pyc $(BIN)
	cd $(BIN); mv API.cpython-38.pyc API.pyc
clean:
	rm -rf  __pycache__

distclean: clean
	rm -f $(BIN)/API.pyc

