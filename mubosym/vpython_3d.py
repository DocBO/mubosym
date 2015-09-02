# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:20:11 2015

@author: oliver
"""

import visual as vis

#import math...

OO = (0.,0.,0.)
OOffset = (0.,-1.5,.0)

def get_v_arg(v,n):
    return v.args[0][0][n]

class PointObject(vis.sphere):
    """
    Visual element point object.
    
    :param state_vec: state vector in the order ???
    :param p: scaling factor
    """
    def __init__(self, state_vec, p, *args, **kwargs):
        super(PointObject, self).__init__(*args, **kwargs)
        self.state_vec = state_vec
        self.n = 0
        self.n_max = len(state_vec[0])
        self.p = p
        #print p
        self.radius = p/6.0
    def set_orient(self,v_orient):
        """
        set orientation to v_orient.
        
        :param v_orient: new orientation of point object
        """
        self.v_orient = v_orient
        
    def update(self):
        """
        update
        """
        global OO
        self.n += 1
        if self.n > self.n_max-1:
            self.n = 0

        self.x = self.state_vec[0][self.n]*self.p - OO[0]
        self.y = self.state_vec[1][self.n]*self.p - OO[1]
        self.z = self.state_vec[2][self.n]*self.p - OO[2]
        
        x1 = self.v_orient[0][self.n]*self.p
        y1 = self.v_orient[1][self.n]*self.p
        z1 = self.v_orient[2][self.n]*self.p
        x2 = self.v_orient[3][self.n]*self.p
        y2 = self.v_orient[4][self.n]*self.p
        z2 = self.v_orient[5][self.n]*self.p
        
        self.axis = (x1,y1,z1)
        self.up = (x2,y2,z2)
        
    def get_pos(self, axes):
        if axes == 'X':
            return (self.state_vec[0][self.n]*self.p+OOffset[0],OOffset[1],OOffset[2])
        elif axes == 'XZ':
            return (self.state_vec[0][self.n]*self.p+OOffset[0],OOffset[1],self.state_vec[2][self.n]*self.p+OOffset[2])
        elif axes == 'all':
            return (self.state_vec[0][self.n]*self.p+OOffset[0],self.state_vec[1][self.n]*self.p+OOffset[1],self.state_vec[2][self.n]*self.p+OOffset[2])
        else:
            return (OOffset[0],OOffset[1],OOffset[2])
            

class BoxObject(vis.box):
    """
    Visual element point object.
    
    :param state_vec: state vector in the order ???
    :param p: scaling factor
    """
    def __init__(self, state_vec, p, length, *args, **kwargs):
        super(BoxObject, self).__init__(*args, **kwargs)
        self.state_vec = state_vec
        self.n = 0
        self.n_max = len(state_vec[0])
        self.p = p
        self.length = length
        #print p
        #self.radius = p/6.0
    def set_orient(self,v_orient):
        """
        set orientation to v_orient.
        
        :param v_orient: new orientation of point object
        """
        self.v_orient = v_orient
        
    def update(self):
        """
        update
        """
        global OO
        self.n += 1
        if self.n > self.n_max-1:
            self.n = 0

        self.x = self.state_vec[0][self.n]*self.p - OO[0]
        self.y = self.state_vec[1][self.n]*self.p - OO[1]
        self.z = self.state_vec[2][self.n]*self.p - OO[2]
        
        x1 = self.v_orient[0][self.n]*self.length
        y1 = self.v_orient[1][self.n]*self.length
        z1 = self.v_orient[2][self.n]*self.length
        x2 = self.v_orient[3][self.n]#*self.p
        y2 = self.v_orient[4][self.n]#*self.p
        z2 = self.v_orient[5][self.n]#*self.p
        
        self.axis = (x1,y1,z1)
        self.up = (x2,y2,z2)
        
    def get_pos(self, axes):
        if axes == 'X':
            return (self.state_vec[0][self.n]*self.p+OOffset[0],OOffset[1],OOffset[2])
        elif axes == 'XZ':
            return (self.state_vec[0][self.n]*self.p+OOffset[0],+OOffset[1],self.state_vec[2][self.n]*self.p+OOffset[2])
        elif axes == 'all':
            return (self.state_vec[0][self.n]*self.p+OOffset[0],self.state_vec[1][self.n]*self.p+OOffset[1],self.state_vec[2][self.n]*self.p+OOffset[2])
        else:
            return (OOffset[0],OOffset[1],OOffset[2])
            
        
class Tire(vis.cylinder):
    """
    Visual element point object.
    
    :param state_vec: state vector in the order ???
    :param p: scaling factor
    """
    def __init__(self, state_vec, p, *args, **kwargs):
        super(Tire, self).__init__(*args, **kwargs)
        self.state_vec = state_vec
        self.n = 0
        self.n_max = len(state_vec[0])
        self.p = p
        #print p
        self.radius = p/3.0
    def set_orient(self,v_orient):
        """
        set orientation to v_orient.
        
        :param v_orient: new orientation of point object
        """
        self.v_orient = v_orient
        
    def update(self):
        """
        update
        """
        global OO
        self.n += 1
        if self.n > self.n_max-1:
            self.n = 0

        

        self.x = self.state_vec[0][self.n]*self.p - OO[0]
        self.y = self.state_vec[1][self.n]*self.p - OO[1]
        self.z = self.state_vec[2][self.n]*self.p - OO[2] #-self.length/2.0
        
        x1 = self.v_orient[0][self.n]#*self.p
        y1 = self.v_orient[1][self.n]#*self.p
        z1 = self.v_orient[2][self.n]#*self.p
        x2 = self.v_orient[6][self.n]#*self.p
        y2 = self.v_orient[7][self.n]#*self.p
        z2 = self.v_orient[8][self.n]#*self.p
        
        self.axis = (x2,y2,z2)
        self.up = (x1,y1,z1)
        s = self.length/2.0
        v_translate = (x2*s, y2*s, z2*s)
        self.x -= v_translate[0]
        self.y -= v_translate[1]
        self.z -= v_translate[2]
        
        
    def get_pos(self, axes):
        if axes == 'X':
            return (self.state_vec[0][self.n]*self.p+OOffset[0],OOffset[1],OOffset[2])
        elif axes == 'XZ':
            return (self.state_vec[0][self.n]*self.p+OOffset[0],+OOffset[1],self.state_vec[2][self.n]*self.p+OOffset[2])
        elif axes == 'all':
            return (self.state_vec[0][self.n]*self.p+OOffset[0],self.state_vec[1][self.n]*self.p+OOffset[1],self.state_vec[2][self.n]*self.p+OOffset[2])
        else:
            return (OOffset[0],OOffset[1],OOffset[2])
            
class Vector_stat(vis.arrow):
    """
    static Vector, it will not be modified.
    
    :param vauf: origin of the vector
    :param v_orient: orientation of the vector
    :param txt: text to be passed to the label
    :param p: scaling of the vector
    """
    def __init__(self, v_auf, v_orient, txt, p, *args, **kwargs):
        super(Vector_stat, self).__init__(*args, **kwargs)
        self.p = p
        if not v_auf == 0:
            x1 = get_v_arg(v_auf,0)*self.p
            y1 = get_v_arg(v_auf,1)*self.p
            z1 = get_v_arg(v_auf,2)*self.p
        else:
            x1 = 0.
            y1 = 0.
            z1 = 0.
        x2 = get_v_arg(v_orient,0)*self.p
        y2 = get_v_arg(v_orient,1)*self.p
        z2 = get_v_arg(v_orient,2)*self.p
        
        self.x = x1
        self.y = y1
        self.z = z1
        self.axis = (x2,y2,z2)
        vis.label(pos=(x1+x2-0.75, y1+y2-0.75,z1+z2-0.75), text=txt)
        
class Vector_dyn(vis.arrow):
    """
    dynamic Vector, provides update function.
    
    :param vauf: origin of the vector
    :param v_orient: orientation of the vector
    :param txt: text to be passed to the label
    :param p: scaling of the vector
    """
    def __init__(self, v_auf, v_orient, txt, p, *args, **kwargs):
        super(Vector_dyn, self).__init__(*args, **kwargs)
        self.n = 0
        self.n_max = len(v_orient[0])
        self.p = p
        self.v_auf = v_auf
        self.v_orient = v_orient
        
    def update(self):
        """
        update
        """
        global OO
        self.n += 1
        if self.n > self.n_max-1:
            self.n = 0
        x1 = self.v_auf[0][self.n]*self.p - OO[0]
        y1 = self.v_auf[1][self.n]*self.p - OO[1]
        z1 = self.v_auf[2][self.n]*self.p - OO[2]
        self.x = x1
        self.y = y1
        self.z = z1
        
        x2 = self.v_orient[0][self.n]*self.p 
        y2 = self.v_orient[1][self.n]*self.p 
        z2 = self.v_orient[2][self.n]*self.p 
        self.axis = (x2,y2,z2)
        #vis.label(pos=(x1+x2-0.75, y1+y2-0.75,z1+z2-0.75), text=txt)

    
class SpringConnection(vis.helix):
    """
    visual representation of spring connection, derived from vis.helix.
    
    :param state_vec: state vector
    :param p:
    """
    def __init__(self, state_vec, p, *args, **kwargs):
        super(SpringConnection, self).__init__(*args, **kwargs)
        self.state_vec = state_vec
        self.n = 0
        self.n_max = len(state_vec[0])
        self.p = p
        self.coils = 12
        self.thickness = 0.2*p/4.0
        self.radius = 0.5*p/4.0
    
    def update(self):
        """
        update to the spring connection
        """
        global OO
        self.n += 1
        if self.n > self.n_max-1:
            self.n = 0
        x1 = self.state_vec[0][self.n]*self.p
        y1 = self.state_vec[1][self.n]*self.p
        z1 = self.state_vec[2][self.n]*self.p
        x2 = self.state_vec[3][self.n]*self.p
        y2 = self.state_vec[4][self.n]*self.p
        z2 = self.state_vec[5][self.n]*self.p
        self.x = x1 - OO[0] 
        self.y = y1 - OO[1] 
        self.z = z1 - OO[2]
        if ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2) < 1e-12:
            return
        self.axis = (x2-x1,y2-y1, z2-z1)
        
        
class RodConnection(vis.cylinder):
    """
    simple rod connection derived from vis.cylinder.
    
    :param state_vec: state vector
    :param p:
    """
    def __init__(self, state_vec, p, myr, *args, **kwargs):
        super(RodConnection, self).__init__(*args, **kwargs)
        self.state_vec = state_vec
        self.n = 0
        self.n_max = len(state_vec[0])
        self.p = p
        #self.coils = 8
        #self.thickness = 0.2*p/4.0
        if myr == 0.:
            self.radius = 0.5*p/8.0
        else:
            self.radius = myr
    def update(self):
        global OO
        self.n += 1
        if self.n > self.n_max-1:
            self.n = 0
        x1 = self.state_vec[0][self.n]*self.p
        y1 = self.state_vec[1][self.n]*self.p
        z1 = self.state_vec[2][self.n]*self.p
        x2 = self.state_vec[3][self.n]*self.p
        y2 = self.state_vec[4][self.n]*self.p
        z2 = self.state_vec[5][self.n]*self.p
        self.x = x1 - OO[0]
        self.y = y1 - OO[1]
        self.z = z1 - OO[2]
        self.axis = (x2-x1,y2-y1, z2-z1)
            
#        #self.image.width = int(math.sqrt((x2-x1)**2+(y2-y1)**2))
        #self.scale = (math.sqrt((x2-x1)**2+(y2-y1)**2))/100.
  


class myLabel(vis.label):
    """
    label class derived from vis.label.
    """
    def __init__(self, txt_vec, *args, **kwargs):
        super(myLabel, self).__init__(*args, **kwargs)
        self.vec = txt_vec
        self.n = 0
        self.n_max = len(txt_vec)
        self.text0 = self.text
    def update(self):
        """
        
        """
        self.n +=1
        if self.n > self.n_max-1:
            self.n = 0
        self.text = self.text0 + '%2.1f'%self.vec[self.n]
        
global game_objects, all_labels

all_labels = []
game_objects = [] 

#
#myScene = vis.display(title='Examples of Tetrahedrons',
#      x=0, y=0, width=600, height=600,
#      center=(5,0,0), background=(0,1,1), range=300)


class animation():
    """
    
    :param state_vec:
    :param con_vec:
    :param con_type:
    :param txt_vec:
    :param dt:
    
    keyword_args:
    
    :param p:
    :param label:
    """
    def __init__(self, scale):
        vis.scene.width = 800
        vis.scene.height = 800
        vis.scene.forward = (-0.2,-0.2,-0.2)
        self.p = scale
        self.center = -1
    
    def s_animation(self, state_vec, orient_vec, con_vec, con_type, bodies_in_graphics, txt_vec, dt, end, speed_factor, p = 20., labels = True, center = -1):
        global game_objects, all_labels
        self.big = bodies_in_graphics
        

        checkerboard = ( (0.2,0.8,0.2,0.8), (0.8,0.2,0.8,0.2), (0.2,0.8,0.2,0.8), (0.8,0.2,0.8,0.2) )
        tex_plane = vis.materials.texture(data=checkerboard,  mapping="rectangular", interpolate=False)
        tex_sphere = vis.materials.texture(data=checkerboard,  mapping="spherical", interpolate=False)
        tex_tire = vis.materials.texture(data=checkerboard,  mapping="rectangular", interpolate=False)
        
        parts = state_vec.shape[1]/3 #assumes 3 Coordinates each timestep each body
        #print parts
        bodies = []
        cons = []
        self.p = p
        
        #self.ball = vis.sphere (pos=(0,4,0), radius=1, material=vis.materials.earth) #material=vis.materials
        #self.ball.velocity = vis.vector(0,-1,0)
        
        for j in range(parts):
            if j in self.big.keys():
                if self.big[j] == 'sphere':
                    bodies.append(PointObject([state_vec[:,j*3],state_vec[:,j*3+1],state_vec[:,j*3+2]], p, pos=(0,0,0), radius=1, material=tex_sphere))
                elif self.big[j] == 'box':
                    #nn = len(bodies)
                    bodies.append(BoxObject([state_vec[:,j*3],state_vec[:,j*3+1],state_vec[:,j*3+2]], p, length = 3.5*p, pos=(0,0,0), height = 0.5*p, width = 2*p))
                elif self.big[j] == 'tire':
                    bodies.append(Tire([state_vec[:,j*3],state_vec[:,j*3+1],state_vec[:,j*3+2]], p, length = 3.5*p, pos=(0,0,0), height = 0.5*p, width = 2*p, color=vis.color.blue, material=tex_tire))
                    nn = len(bodies)
            else:
                bodies.append(PointObject([state_vec[:,j*3],state_vec[:,j*3+1],state_vec[:,j*3+2]], p, pos=(0,0,0), radius=1, material=tex_sphere))
            bodies[-1].set_orient([orient_vec[:,j*9],orient_vec[:,j*9+1],orient_vec[:,j*9+2],
                                   orient_vec[:,j*9+3],orient_vec[:,j*9+4],orient_vec[:,j*9+5],
                                   orient_vec[:,j*9+6],orient_vec[:,j*9+7],orient_vec[:,j*9+8]])

        

        self.tau = 0.
        self.dt = dt
#        if labels:
#            self.timer = vis.label(pos=(0,0,0), text='Time: %2.1f' % self.tau)
        
        for j in range(parts):
            #print con_type[j]
            if con_type[j] == 'transparent':
                pass
                #cons.append(RodConnection([con_vec[:,j*6],con_vec[:,j*6+1],con_vec[:,j*6+2],con_vec[:,j*6+3],con_vec[:,j*6+4],con_vec[:,j*6+5]], p, 0.4, pos=(0,0,0), axis=(5,0,0), opacity = 0.2 ))
            elif not con_type[j] == 'y-axes':
                cons.append(RodConnection([con_vec[:,j*6],con_vec[:,j*6+1],con_vec[:,j*6+2],con_vec[:,j*6+3],con_vec[:,j*6+4],con_vec[:,j*6+5]], p, 0.0, pos=(0,0,0), axis=(5,0,0), material=vis.materials.wood))
            else:
                cons.append(SpringConnection([con_vec[:,j*6],con_vec[:,j*6+1],con_vec[:,j*6+2],con_vec[:,j*6+3],con_vec[:,j*6+4],con_vec[:,j*6+5]], p,pos=(0,0,0), axis=(5,0,0), radius=0.3))
                
        #print "p: ",p
        if labels:
            all_labels.append(myLabel(txt_vec, pos=(0,p/2.0,0), text='Velocity [m/s]: '))
        
        r = 1.0
        self.floor = vis.box(axis=(0,1,0), length=0.5, height=r*20*p/4.0, width=r*20*p/4.0, color=vis.color.cyan, material=tex_plane, opacity=0.7)
        if center > -1:
            self.center = center + len(game_objects)
            #print self.center, len(game_objects)
        game_objects += bodies + cons
        
        self.start(end, speed_factor)
        return vis.scene
                        
    def set_stationary_vectors(self, vs):
        for v in vs:
            print v
            Vector_stat(v[0], v[1], '', self.p, pos=(0,0,0), axis=(5,0,0), shaftwidth=0.2)
    
    def set_stationary_frame(self, mf):
        orig = mf.get_pos_IF()
        ex = mf.get_ex_IF()
        ey = mf.get_ey_IF()
        ez = mf.get_ez_IF()
        Vector_stat(orig, ex, 'x', self.p, pos=(0,0,0), axis=(5,0,0), shaftwidth=0.3)
        Vector_stat(orig, ey, 'y', self.p, pos=(0,0,0), axis=(5,0,0), shaftwidth=0.3)
        Vector_stat(orig, ez, 'z', self.p, pos=(0,0,0), axis=(5,0,0), shaftwidth=0.3)
            
    def set_dynamic_frame(self, frame_vec):
        global game_objects
        orig = [frame_vec[:,0],frame_vec[:,1],frame_vec[:,2]]
        ex = [frame_vec[:,3],frame_vec[:,4],frame_vec[:,5]]
        ey = [frame_vec[:,6],frame_vec[:,7],frame_vec[:,8]]
        ez = [frame_vec[:,9],frame_vec[:,10],frame_vec[:,11]]
        vectors = []
        vectors.append(Vector_dyn(orig, ex, 'x', self.p, pos=(0,0,0), axis=(5,0,0), shaftwidth=0.3))
        vectors.append(Vector_dyn(orig, ey, 'y', self.p, pos=(0,0,0), axis=(5,0,0), shaftwidth=0.3))
        vectors.append(Vector_dyn(orig, ez, 'z', self.p, pos=(0,0,0), axis=(5,0,0), shaftwidth=0.3))
        game_objects += vectors
        
    def set_force(self, force, scale=1e-1, f_min = 0.1, f_max = 10.):
        global game_objects
        orig = [force[:,0],force[:,1],force[:,2]]
        ff = [force[:,3]*scale,force[:,4]*scale,force[:,5]*scale]
        for ii in range(len(ff[0])):
            magn = vis.sqrt(ff[0][ii]*ff[0][ii]+ff[1][ii]*ff[1][ii]+ff[2][ii]*ff[2][ii])+1e-3
            if magn > f_max:
                ff[0][ii] = ff[0][ii] * f_max/magn
                ff[1][ii] = ff[1][ii] * f_max/magn
                ff[2][ii] = ff[2][ii] * f_max/magn
            if magn < f_min:
                ff[0][ii] = ff[0][ii] * f_min/magn
                ff[1][ii] = ff[1][ii] * f_min/magn
                ff[2][ii] = ff[2][ii] * f_min/magn
        
        v = Vector_dyn(orig, ff, 'F', self.p, pos=(0,0,0), axis=(5,0,0), shaftwidth=0.5, color=vis.color.red)
        game_objects.append( v )
        
    
    def start(self, end, speed_factor):
        global game_objects, OO
        while 1:
            vis.rate (1/self.dt*speed_factor)
            self.tau += self.dt
            #vis.scene.autoscale = False
            #vis.scene.range = (-4.,-4.,-4.)
            if self.center > -1:
                #print type(game_objects[self.center])
                OO = game_objects[self.center].get_pos('XZ') 
                #vis.scene.center = game_objects[self.center].get_pos('XZ') 
            if self.tau > end:
                return
            for obj in game_objects:
                obj.update()
            for label in all_labels:
                label.update()
        
            
                
                        
            
