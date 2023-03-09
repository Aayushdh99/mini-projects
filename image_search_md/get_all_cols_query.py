import sqlite3

connection = sqlite3.connect('imagemetadata.db')
cursor = connection.execute('SELECT * FROM IMAGE_TAGS')
names = list(map(lambda x: x[0], cursor.description))
connection.close()
print(names)