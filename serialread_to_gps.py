import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time
import serial


data_gelen = serial.Serial("COM6", baudrate=9600)
fig=plt.figure()

ax1=fig.add_subplot(2,1,1)
ax2=fig.add_subplot(2,1,2)
x1s=[]
y1s=[]
x2s=[]
y2s=[]
def data(i):

    dataayrı = data_gelen.readline()
    data = str(dataayrı).split("'")
    data_tam = data[1]
    data_dizi = data_tam.split(",")
    if data_dizi[0]=="gas":
        print(data_dizi[1][0:2])
        ys1 = int(data_dizi[1][0:2])
        y1s.append(ys1)
        x1s.append(time.time())
    if data_dizi[0] == "sic":
        print(data_dizi[1][0:2])
        ys2 = int(data_dizi[1][0:2])
        y2s.append(ys2)
        x2s.append(time.time())
    if data_dizi[0]=="$GPRMC":
        if data_dizi[2]=="A":

            gpslat =float(data_dizi[3])
            if data_dizi[4]=="S":
                gpslat=-gpslat
            latdeg =int(gpslat/100)
            latmin = gpslat - latdeg*100
            lat = latdeg+(latmin/60)
            longps=float(data_dizi[5])

            if data_dizi[6] =="W":
                longps=-longps
            londeg=int(longps/100)
            lonmin=longps-londeg*100
            lon=londeg+(lonmin/60)

            with open("position.kml","w") as pos:
                pos.write("""<kml xmlns="http://www.opengis.net/kml/2.2"
     xmlns:gx="http://www.google.com/kml/ext/2.2"><Placemark>
      <name>HIDROANA</name>
      <description></description>
      <Point>
        <coordinates>%s,%s,0</coordinates>
      </Point>
    </Placemark></kml>"""% (lon,lat))


    ax1.plot(x1s,y1s)
    ax2.plot(x2s,y2s)

ani=animation.FuncAnimation(fig,data,interval=50)
plt.show()

