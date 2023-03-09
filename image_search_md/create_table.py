import sqlite3

# Connecting to sqlite
# connection object
connection_obj = sqlite3.connect('imagemetadata.db')

# cursor object
cursor_obj = connection_obj.cursor()

# Drop the GEEK table if already exists.
# cursor_obj.execute("DROP TABLE IF EXISTS IMAGE_TAGS")

# # Creating table
# table = """ CREATE TABLE IMAGE_TAGS (
# 			Email VARCHAR(255) NOT NULL,
# 			First_Name CHAR(25) NOT NULL,
# 			Last_Name CHAR(25),
# 			Score INT
# 		); """
# Creating table
table = """ CREATE TABLE IMAGE_TAGS (
            image_name VARCHAR(255) NOT NULL,
            person INT,
            bicycle INT,
            car INT,
            motorcycle INT,
            airplane INT,
            bus INT,
            train INT,
            truck INT,
            boat INT,
            traffic INT,
            fire INT,
            street INT,
            stop INT,
            parking INT,
            bench INT,
            bird INT,
            cat INT,
            dog INT,
            horse INT,
            sheep INT,
            cow INT,
            elephant INT,
            bear INT,
            zebra INT,
            giraffe INT,
            hat INT,
            backpack INT,
            umbrella INT,
            shoe INT,
            eye INT,
            handbag INT,
            tie INT,
            suitcase INT,
            frisbee INT,
            skis INT,
            snowboard INT,
            sports INT,
            kite INT,
            baseball INT,
            skateboard INT,
            surfboard INT,
            tennis INT,
            bottle INT,
            plate INT,
            wine INT,
            cup INT,
            fork INT,
            knife INT,
            spoon INT,
            bowl INT,
            banana INT,
            apple INT,
            sandwich INT,
            orange INT,
            broccoli INT,
            carrot INT,
            hot INT,
            pizza INT,
            donut INT,
            cake INT,
            chair INT,
            couch INT,
            potted INT,
            bed INT,
            mirror INT,
            dining INT,
            window INT,
            desk INT,
            toilet INT,
            door INT,
            tv INT,
            laptop INT,
            mouse INT,
            remote INT,
            keyboard INT,
            cell INT,
            microwave INT,
            oven INT,
            toaster INT,
            sink INT,
            refrigerator INT,
            blender INT,
            book INT,
            clock INT,
            vase INT,
            scissors INT,
            teddy INT,
            toothbrush INT,
            hair INT
        ); """

cursor_obj.execute(table)

print("Table is Ready")

# Close the connection
connection_obj.close()
