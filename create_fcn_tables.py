# creates the tables for a new fcn database to store data for later analysis.

import psycopg2

conn = psycopg2.connect("dbname=fcn_rgb user=dnn_user password=pgpswd")

cur = conn.cursor()

cur.execute("CREATE TABLE experiment (id bigserial PRIMARY KEY, "
            "created_date timestamp NOT NULL DEFAULT NOW(), "
            "name varchar(50) NOT NULL, description varchar(255));")

cur.execute("CREATE TABLE losses (id bigserial PRIMARY KEY, "
            "experiment_id BIGINT REFERENCES experiment (id), "
            "created_date TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(), "
            "epoch INT NOT NULL, iteration INT NOT NULL, "
            "loss DOUBLE PRECISION NOT NULL, "
            "training BOOLEAN NOT NULL, "
            "image varchar(255) NOT NULL, flip BOOLEAN NOT NULL, "
            "rotation REAL NOT NULL, size_idx INT NOT NULL, "
            "cut_x INT NOT NULL, cut_y INT NOT NULL);")

cur.execute("CREATE TABLE accuracies (id bigserial PRIMARY KEY, "
            "experiment_id BIGINT REFERENCES experiment (id), "
            "created_date TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(), "
            "epoch INT NOT NULL, iteration INT NOT NULL, "
            "training BOOLEAN NOT NULL, "
            "accuracy DOUBLE PRECISION NOT NULL, "
            "image varchar(255) NOT NULL, flip BOOLEAN NOT NULL, "
            "rotation REAL NOT NULL, size_idx INT NOT NULL, "
            "cut_x INT NOT NULL, cut_y INT NOT NULL);")


# cur.execute("INSERT INTO test (num, data) VALUES (%s, %s)", (100, "abc'def"))

# cur.execute("SELECT * FROM test;")

# print(cur.fetchone())

conn.commit()

cur.close()
conn.close()

