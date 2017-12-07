import pymysql

try:
    db = pymysql.connect(ip_adress,id_,pswd,"hidroana" )
except Exception as e :
    print (e)
cursor = db.cursor()
print("bağlandı")
cursor.execute("SELECT VERSION()")

#cursor.execute("DROP TABLE IF EXISTS TELEMETRI")
sql = """CREATE TABLE TELEMETRI (
   id  INT NOT NULL,
   LAT FLOAT,
   LON FLOAT,
   SPEED FLOAT,
   SICAKLIK FLOAT,
   GAZ FL OAT)"""

cursor.execute(sql)
print("tablo oluşturuldu")

