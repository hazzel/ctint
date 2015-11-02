import matplotlib
matplotlib.use('TkAgg') 
import pylab
import glob

list_of_files = glob.glob("../data/ed*")
datalist = [ ( pylab.loadtxt(filename), label ) for label, filename in enumerate(list_of_files) ]

for data, label in datalist:
	#pylab.figure()
	pylab.plot( data[:,3], data[:,5], label="m2", linestyle="-", color="b", linewidth=2.0 )
	pylab.plot( data[:,3], data[:,6], label="m4", linestyle="-", color="r", linewidth=2.0 )
	#pylab.plot( data[:,3], data[:,7], label="B", linestyle="-", color="g", linewidth=2.0 )

pylab.legend()
pylab.title("Title of Plot")
pylab.xlabel("X Axis Label")
pylab.ylabel("Y Axis Label")
pylab.show()