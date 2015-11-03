import matplotlib
matplotlib.use('TkAgg') 
import pylab
import glob

list_of_files = glob.glob("../data/V2.0/ed*")
datalist = [ ( pylab.loadtxt(filename), label ) for label, filename in enumerate(list_of_files) ]

for data, label in datalist:
	#pylab.figure()
	L = int(data[:,1][0])
	if L == 1:
		c = ["c", "m"]
	elif L == 2:
		c = ["b", "g"]
	elif L == 3:
		c = ["r", "k"]
	pylab.plot( data[:,3]*L, data[:,5]*L**1., label="L="+str(L)+", m2", linestyle="-", color=c[0], linewidth=2.0 )
	#pylab.plot( data[:,3]*L, data[:,6], label="L="+str(L)+", m4", linestyle="-", color=c[1], linewidth=2.0 )
	#pylab.plot( data[:,3]*L, data[:,7], label="L="+str(L)+", B", linestyle="-", color=c[0], linewidth=2.0 )

pylab.legend()
pylab.title("Title of Plot")
pylab.xlabel("X Axis Label")
pylab.ylabel("Y Axis Label")
pylab.show()