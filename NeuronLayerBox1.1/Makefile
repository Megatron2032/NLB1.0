NLB : SM1-0.cu
	#gcc -D_REENTRANT qt-show.c -o show -lpthread -lglut -lGLU -lGL
	nvcc -D_REENTRANT -o NLB SM1-0.cu
	gcc -fPIC -shared Interface/c/init.c Interface/c/input.c Interface/c/output.c -o libNLB.so
clean:
	rm -rf *.o NLB
	rm -rf NLB
	#rm -rf show
	@echo "Clean done!"
