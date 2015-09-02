# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 08:34:00 2015

@author: loc_yangtingting
"""

import sqlite3
               
name = '/Datenmodell.db'

class dbHandler(object):
    def __init__(self, path):
        '''
        dbhandler object
        '''
        print "try to open: "+path+name
        self.conn = sqlite3.connect(path+name)
        self.c = self.conn.cursor()
        sql = 'CREATE TABLE if not exists models (name text PRIMARY KEY, expr text)'
        self.c.execute(sql)
        self.conn.commit()
        self.get_model_names()
        
    def drop_table(self):
        self.c.execute('DROP TABLE if exists models')
        self.conn.commit()
    
    def get_model_names(self):
        self.c.execute('SELECT * FROM models ')
        rows = self.c.fetchall()
        self.keys = [r[0] for r in rows]
        return rows
        
    def close(self):
        self.conn.close()
                
    def get(self, key):
        self.c.execute('SELECT * FROM models WHERE name=?', (key,))
        row = self.c.fetchone()
        return row
            
    def put(self, key, obj_str):
        if self.has_key(key):
            self.c.execute('DELETE FROM models WHERE name=?', (key,))
        self.c.execute('INSERT INTO models (name, expr) values (?, ?)', (key, obj_str))
        self.conn.commit()
        
    def has_key(self,k):
        return k in self.keys
    
        
