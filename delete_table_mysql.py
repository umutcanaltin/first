import pymysql
db = pymysql.connect(ip_adress,id_,pswd,"hidroana" )
cursor = db.cursor()
print("bağlandı")
cursor.execute("SELECT VERSION()")

cursor.execute("DROP TABLE IF EXISTS TELEMETRI")

print("tablo silindi")

