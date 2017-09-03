# -*- coding: utf-8 -*-
"""
Created on Thu Sep 03 14:41:24 2015

@author: ecksjoh, oliver

this module connects to vispy
"""
import os,sys
import numpy as np
from vispy import app,gloo,scene,visuals
from vispy.scene.visuals import create_visual_node
from vispy.visuals.transforms import STTransform, MatrixTransform, ChainTransform
from vispy.visuals.filters import Alpha
#from vispy.io import read_mesh,load_data_file,load_crate

#from vispy.util.transforms import perspective,translate,rotate

OO = (0.,0.,0.)
OOffset = (0.,-1.5,.0)
global game_objects, all_labels

all_labels = []
game_objects = []

class Body():
    """
    binds tranfos to the bodies and the order of doing it
    """
    def __init__(self, state_vec, p):
        self.state_vec = state_vec
        self.n = 0
        self.n_max = len(state_vec)
        self.p = p
        self.radius = p/6.0

        self.rot = MatrixTransform()
        self.v_orient = None
        self.x = 0.
        self.y = 0.
        self.z = 0.
        #print("-------", self.n_max, self.p)

    def set_orient(self,v_orient):
        """
        set orientation to v_orient.

        :param v_orient: new orientation of point object
        """
        self.v_orient = v_orient

    def _update(self):
        """
        update

        rotation and then translation
        """
        global OO
        self.n += 1
        if self.n > self.n_max-1:
            self.n = 0
            return
        
        dx = self.state_vec[self.n][0] 
        dy = self.state_vec[self.n][1] 
        dz = self.state_vec[self.n][2]
            
        if self.v_orient:
            new_base = self.v_orient[self.n]
        else:
            new_base = None

        self.trafo(dx,dy,dz,base=new_base)

    def _norm(self, n):
        """ normalizing a vector n = (x,y,z) """
        x, y, z = n
        norm = np.sqrt(x*x + y*y + z*z)
        return x/norm, y/norm, z/norm

    def _cross(self, n0, n1):
        """ doing a crossproduct i.e. n1 x n2 """
        x0, y0, z0 = n0
        x1, y1, z1 = n1
        x = y0*z1 - z0*y1
        y = z0*x1 - x0*z1
        z = x0*y1 - y0*x1
        return x,y,z

    def _ortho(self, n):
        """ finding an arbitrary orthogonal vector to another in 3d """
        x,y,z = n
        if z!=0. and y!=0.:
            return 0., z, -y
        elif x!=0. and y!=0.:
            return y, -x, 0.
        elif x!=0. and z!=0.:
            return z, 0., -x
        else:
            return x,y,z+1

    def _get_ortho_base(self, n):
        """ 
        calc an ottho base for one direction, such that ex is pointing in the end to that direction 

        """
        ex = self._norm(n)        
        ey = self._norm(self._ortho(ex))
        ez = self._cross(ex, ey)
        return ex, ey, ez


    def trafo(self, x=0.,y=0.,z=0., base=None, scale=None, reset=True):
        """
        doing first the scale then rotation and afterwards translate
        """
        if reset:
            self.rot.reset()
        if scale is not None:
            self.rot.scale(scale)
        if base is not None:
            self.rot.mult_rotation(base)
        self.rot.translate((x,y,z))
        self.transform = self.rot


class mbVector(scene.visuals.Vector, Body):
    #rows, cols, radius, length, cone_radius, cone_length, vertex_colors=None, face_colors=None,
    #             color=(0.5, 0.5, 1, 1), edge_color=None
    def __init__(self, view, face_color, state_vec, p):
        super(mbVector, self).__init__(10, 10, 0.05, 1., 0.1, 0.25, color=face_color, shading=None, parent=view)
        self.unfreeze()
        Body.__init__(self, state_vec, p)

    def _update(self):
        """
        update

        includes the scale information, handles the rotation differently
        """
        global OO
        self.n += 1
        if self.n > self.n_max-1:
            self.n = 0
            return
           
        self.rot.set_rotation((0,0,1,0,1,0,1,0,0))
        scale_x = self.v_orient[self.n][0]
        scale_y = self.v_orient[self.n][1] 
        scale_z = self.v_orient[self.n][2]
        scale = np.sqrt(scale_x*scale_x+scale_y*scale_y+scale_z*scale_z)
        
        scale = (scale, 1., 1.)

        dx = self.state_vec[self.n][0]
        dy = self.state_vec[self.n][1] 
        dz = self.state_vec[self.n][2]
        
        ex, ey, ez = self._get_ortho_base((scale_x, scale_y, scale_z))
        new_base = (*ex,*ey,*ez)

        self.trafo( dx,dy,dz, base=new_base , scale=scale, reset=False)
        

class mbTube(scene.visuals.Tube, Body):
    """
    creates a custom tube:
        create_visual_node(visuals.TubeVisual)
        
    :param view: the view as obtained by call to e.g. scene.SceneCanvas().central_widget.add_view()
    :param path: An array of (x, y, z) points describing the path along which the tube will be extruded.
    :param radius: the radius
    :param face_color: the faces color
    :param tube_points: The number of points in the circle, default is 16

    """
    def __init__(self,view,radius,path,face_color,state_vec, p, tube_points=16, closed=False):
        """        
        points : ndarray
            An array of (x, y, z) points describing the path along which the
            tube will be extruded.
        radius : float
            The radius of the tube. Defaults to 1.0.
        closed : bool
            Whether the tube should be closed, joining the last point to the
            first. Defaults to False.
        color : Color | ColorArray
            The color(s) to use when drawing the tube. The same color is
            applied to each vertex of the mesh surrounding each point of
            the line. If the input is a ColorArray, the argument will be
            cycled; for instance if 'red' is passed then the entire tube
            will be red, or if ['green', 'blue'] is passed then the points
            will alternate between these colours. Defaults to 'purple'.
        tube_points : int
            The number of points in the circle-approximating polygon of the
            tube's cross section. Defaults to 8.
        shading : str | None
            Same as for the `MeshVisual` class. Defaults to 'smooth'.
        vertex_colors: ndarray | None
            Same as for the `MeshVisual` class.
        face_colors: ndarray | None
            Same as for the `MeshVisual` class.
        mode : str
            Same as for the `MeshVisual` class. Defaults to 'triangles'.
        """
        super(mbTube, self).__init__(radius=radius,points=path,color=face_color, shading="smooth",tube_points=tube_points,closed=closed,parent=view)
        self.unfreeze()
        Body.__init__(self, state_vec, p)

    def _update(self):
        """
        update

        includes the scale information, handles the rotation differently
        """
        global OO
        self.n += 1
        if self.n > self.n_max-1:
            self.n = 0
            return
        
        scale_x = 1.*(self.state_vec[self.n][3] - self.state_vec[self.n][0])
        scale_y = 1.*(self.state_vec[self.n][4] - self.state_vec[self.n][1])
        scale_z = 1.*(self.state_vec[self.n][5] - self.state_vec[self.n][2])
        scale = np.sqrt(scale_x*scale_x+scale_y*scale_y+scale_z*scale_z)
        scale = (scale, 1., 1.)

        dx = self.state_vec[self.n][0]
        dy = self.state_vec[self.n][1] 
        dz = self.state_vec[self.n][2]

        ex, ey, ez = self._get_ortho_base((scale_x, scale_y, scale_z))
        new_base = (*ex,*ey,*ez)

        self.trafo( dx,dy,dz, base=new_base , scale=scale)


class mbFrame(scene.visuals.XYZAxis, Body):
    """
    creates a custom Body Frame, a 3D axis for indicating coordinate system orientation. Axes are x=red, y=green, z=blue.
        create_visual_node(visuals.XYZAxisVisual)

    maybee this will become the subvisual in every mbVisual

    :param view: the view as obtained by call to e.g. scene.SceneCanvas().central_widget.add_view()

    """
    def __init__(self, view, state_vec, p):
        super(mbFrame, self).__init__(parent=view)
        self.unfreeze()
        Body.__init__(self, state_vec, p)

class mbCube(scene.visuals.Cube, Body):
    """
    creates a custom cube:
        create_visual_node(visuals.CubeVisual)

    :param view: the view as obtained by call to e.g. scene.SceneCanvas().central_widget.add_view()
    :param a: first edge lenght (in x direction)
    :param b: second edge lenght (in y direction)
    :param c: third edge lenght (in z direction)
    :param face_color: the faces color
    :param edge_color: the edge (wire) color

    """
    def __init__(self,view,a,b,c,face_color,edge_color, state_vec, p):
        super(mbCube,self).__init__((a,b,c),color=face_color,edge_color=edge_color,parent=view, shading="smooth")
        self.unfreeze()
        Body.__init__(self, state_vec, p)


class mbSphere(scene.visuals.Sphere, Body):
    """
    creates a custom sphere:
        create_visual_node(visuals.SphereVisual)

    :param view: the view as obtained by call to e.g. scene.SceneCanvas().central_widget.add_view()
    :param radius: the radius
    :param face_color: the faces color
    :param edge_color: the edge (wire) color
    
    """
    def __init__(self,view,radius,face_color,edge_color, state_vec, p):
        super(mbSphere,self).__init__(radius,color=face_color,edge_color="grey",parent=view,method='ico', subdivisions=2 )
        self.unfreeze()
        Body.__init__(self, state_vec, p)


class mbCanvas(scene.SceneCanvas):
    def __init__(self):
        super(mbCanvas, self).__init__( keys='interactive', bgcolor='white',
                           size=(800, 600), show=True)

        self.unfreeze()
        self.tt = 0.
        self.timer = app.Timer(interval=1e-3) #or use 'auto'
        #print(timer.interval)
        self.timer.connect(self.on_timer)
        
        #adds a viewbox to the canvas-central-widget
        self.view = self.central_widget.add_view()
        #call the setter for the camera
        self.view.camera = 'arcball'
    
        self.view.camera.set_range(x=[-6, 6])
        gloo.set_state(blend=True)
    # ---------------------------------
    def on_key_press(self, event):
        """
        is included in the canvas. here used to start/stop the animation loop
        """
        if event.text == ' ':
            if self.timer.running:
                self.timer.stop()
            else:
                self.timer.start()

    def on_mouse_press(self, event):
        print(event.pos)

    # ---------------------------------
    def on_timer(self, event):
        """
        the update frame function, for physical processes must consider the global time
        
        called with interval if possible or mostly in arbitrary time steps
        """
        global game_objects, all_labels
        #print("TIMER ",self.tt, game_objects)
        self.tt += event.dt
        for obj in game_objects:
            obj._update()
        for label in all_labels:
            label._update()
        # the following is not clear:        
        # self.update()
    

class mbText(scene.visuals.Text, Body):
    # text=None, color='black', bold=False,
    #             italic=False, face='OpenSans', font_size=12, pos=[0, 0, 0],
    #             rotation=0., anchor_x='center', anchor_y='center',
    #             font_manager=None)
    def __init__(self, view, text, state_vec, p):
        super(mbText,self).__init__(text, bold=True, parent=view)
        self.unfreeze()
        Body.__init__(self, state_vec, p)



class animation():
    """
    this is the main visualization class

    trying to tidy up global game_objects, all_labels....
    """
    def __init__(self, scale=1.):
        self.canvas = mbCanvas()
        self.p = scale
        self.scene = self.canvas.view.scene

    def start_animation(self, body_names,state_vec,orient_vec,con_vec,con_type,bodies_in_graphics,txt_vec,dt,end,speed_factor,p=1.,labels=True,center=-1):
        """
        :param state_vec: 3 coordinates each body
        :param orient_vec: orientation vector
        :param con_vec:  connection vector
        :param con_type: connection type
        :param bodies_in_graphics: dictionary of bodies types
        :param txt_vec: text vector for labels
        :param dt: time step increment
        :param end: stop mark for visualization
        :param speed_factor: factor scale dt, time step increment

        keyword_args:

        :param p:
        :param label:
        """
        global game_objects, all_labels
        self.big = bodies_in_graphics

        #checkerboard = ( (0.2,0.8,0.2,0.8), (0.8,0.2,0.8,0.2), (0.2,0.8,0.2,0.8), (0.8,0.2,0.8,0.2) )
        #tex_plane = vis.materials.texture(data=checkerboard,  mapping="rectangular", interpolate=False)
        #tex_sphere = vis.materials.texture(data=checkerboard,  mapping="spherical", interpolate=False)
        #tex_tire = vis.materials.texture(data=checkerboard,  mapping="rectangular", interpolate=False)

        parts = int(state_vec.shape[1]/3) #assumes 3 Coordinates each timestep each body
        #print parts
        bodies = []
        cons = []
        self.p = p

        #self.ball = vis.sphere (pos=(0,4,0), radius=1, material=vis.materials.earth) #material=vis.materials
        #self.ball.velocity = vis.vector(0,-1,0)

        for j in range(parts):
            state_vec_ = [(x[j*3],x[j*3+2],x[j*3+1]) for x in state_vec]
            if j in self.big.keys():
                
                if self.big[j] == 'sphere':
                    bodies.append(mbSphere(self.scene,1,'blue','', state_vec_, 1.))
                elif self.big[j] == 'box':
                    bodies.append(mbCube(self.scene,0.4,0.4,0.4,'red','red', state_vec_, 1.))
                elif self.big[j] == 'tire':
                    bodies.append(mbSphere(self.scene,1,'blue','', state_vec_, 1.))                   
                    #bodies.append(Tire([state_vec[:,j*3],state_vec[:,j*3+1],state_vec[:,j*3+2]], p, length = 3.5*p, pos=(0,0,0), height = 0.5*p, width = 2*p, color=vis.color.blue, material=tex_tire))
                    #nn = len(bodies)
            else:
                bodies.append(mbSphere(self.scene, 0.3, 'blue', '', state_vec_, 1.))
                all_labels.append(mbText(self.scene, body_names[j], state_vec_, 1.))
            orient_vec_ = [ (x[j*9],x[j*9+2],x[j*9+1],x[j*9+3],x[j*9+5],x[j*9+4],x[j*9+6],x[j*9+8],x[j*9+7]) for x in orient_vec]
            bodies[-1].set_orient(orient_vec_)

        self.tau = 0.
        self.dt = dt
#        if labels:
#            self.timer = vis.label(pos=(0,0,0), text='Time: %2.1f' % self.tau)

        for j in range(parts):
            if con_type[j] == 'transparent':
                cons.append(mbTube(self.scene,0.1,[(0.,0.,0.),(1.,0.,0.)],"blue",con_vec_, self.p))
                cons[-1].attach(Alpha(0.2))
            elif not con_type[j] == 'y-axes':
                con_vec_ = [ (x[j*6],x[j*6+2],x[j*6+1],x[j*6+3],x[j*6+5],x[j*6+4]) for x in con_vec]
                cons.append(mbTube(self.scene,0.1,[(0.,0.,0.),(1.,0.,0.)],"blue",con_vec_, self.p))
            else:
                pass
                #cons.append(SpringConnection([con_vec[:,j*6],con_vec[:,j*6+1],con_vec[:,j*6+2],con_vec[:,j*6+3],con_vec[:,j*6+4],con_vec[:,j*6+5]], p,pos=(0,0,0), axis=(5,0,0), radius=0.3))

        #print "p: ",p
        if labels:
            pass
            #all_labels.append(myLabel(txt_vec, pos=(0,p/2.0,0), text='Velocity [m/s]: '))

        r = 1.0
        self.floor = mbCube(self.scene,10,10,0.1,'red','red', [], 0)
        self.floor.attach(Alpha(0.2))
        #self.floor = vis.box(axis=(0,1,0), length=0.5, height=r*20*p/4.0, width=r*20*p/4.0, color=vis.color.cyan, material=tex_plane, opacity=0.7)
        #if center > -1:
        #    self.center = center + len(game_objects)
            #print self.center, len(game_objects)
        game_objects += bodies  + cons

        self.start(end, speed_factor)
        return

    def set_stationary_vectors(self, vs):
        """

        :param vs: stationary vectors
        """
        return
        for v in vs:
            Vector_stat(v[0], v[1], '', self.p, pos=(0,0,0), axis=(5,0,0), shaftwidth=0.2)

    def set_stationary_frame(self, mf):
        """

        :param mf: stationary frames
        """
        return
        orig = mf.get_pos_IF()
        ex = mf.get_ex_IF()
        ey = mf.get_ey_IF()
        ez = mf.get_ez_IF()
        Vector_stat(orig, ex, 'x', self.p, pos=(0,0,0), axis=(5,0,0), shaftwidth=0.3)
        Vector_stat(orig, ey, 'y', self.p, pos=(0,0,0), axis=(5,0,0), shaftwidth=0.3)
        Vector_stat(orig, ez, 'z', self.p, pos=(0,0,0), axis=(5,0,0), shaftwidth=0.3)

    def set_dynamic_frame(self, frame_vec):
        """

        :param frame_vec: vector containing informations for dynamic frame
        0,1,2 origin
        3,4,5 first elementary axis
        6,7,8 second...
        9,10,11 third...
        """
        global game_objects
        state_vec_ = [(x[0],x[2],x[1]) for x in frame_vec]
        game_objects.append(mbFrame(self.scene, state_vec_, 1.))
        orient_vec_ = [(x[3],x[5],x[4],x[6],x[8],x[7],x[9],x[11],x[10]) for x in frame_vec]
        game_objects[-1].set_orient(orient_vec_)
        

    def set_force(self, force, scale=1e-1, f_min = 0.1, f_max = 10.):
        """

        :param force: 3*2 states force
        :param scale: scaling in velocity
        :param f_min: minimal magnitude of force
        :param f_max: maximal magnitude of force
        """
        global game_objects
        orig = [(x[0],x[2],x[1]) for x in force]
        full_vec = [[x[3]*scale,x[5]*scale,x[4]*scale] for x in force]
        
        for ff in full_vec:
            magn = np.sqrt(ff[0]*ff[0]+ff[1]*ff[1]+ff[2]*ff[2])+1e-3
            if magn > f_max:
                ff[0] = ff[0] * f_max/magn
                ff[1] = ff[1] * f_max/magn
                ff[2] = ff[2] * f_max/magn
            if magn < f_min:
                ff[0] = ff[0] * f_min/magn
                ff[1] = ff[1] * f_min/magn
                ff[2] = ff[2] * f_min/magn
        v = mbVector(self.scene, "red", orig, 1.)
        game_objects.append( v )
        game_objects[-1].set_orient(full_vec)



    def start(self, end, speed_factor):
        """

        :param end: end of visualization, should be less or equal then integration frame end.
        :param speed_factor: factor for visualization speed
        """
        
        self.canvas.timer.start(0)
        self.canvas.measure_fps()
        self.canvas.app.run()
        #vis.scene.autoscale = False
        #vis.scene.range = (-4.,-4.,-4.)
        #if self.center > -1:
        #    #print type(game_objects[self.center])
        #    OO = game_objects[self.center].get_pos('XZ')
        #    #vis.scene.center = game_objects[self.center].get_pos('XZ')
            



if __name__ == '__main__' and sys.flags.interactive == 0:
    canvas = mbCanvas() 


    body_frame = mbFrame(canvas.view.scene, [], 0)
    #cube = mbCube(canvas.view.scene,2,3,1,'red','red', [(1,0,0),(1,0,0)], 0)
    #sphere = mbSphere(canvas.view.scene,2,'blue','', [(0,0,1)], 0)
    #tube = mbTube(canvas.view.scene,2.5,[(0.,0.,0.),(2.,0.,0.)],"green",[(0,0,0,0,0,0),(2,2,2,4,4,4)], 0)
    #cube._update()
    #cube._update()
    #sphere._update()
    #tube._update()
    #arrow = mbVector(canvas.view.scene, "red", [], 0)
    text = mbText(canvas.view.scene, "Knack", [(0,0,1),(0,0,1)], 0)
    text._update()
    canvas.app.run()
    

#Volume = create_visual_node(visuals.VolumeVisual)
#Arrow = create_visual_node(visuals.ArrowVisual)
#Axis = create_visual_node(visuals.AxisVisual)
#Box = create_visual_node(visuals.BoxVisual)
#ColorBar = create_visual_node(visuals.ColorBarVisual)
#Compound = create_visual_node(visuals.CompoundVisual)
#Ellipse = create_visual_node(visuals.EllipseVisual)
#GridLines = create_visual_node(visuals.GridLinesVisual)
#Histogram = create_visual_node(visuals.HistogramVisual)
#Image = create_visual_node(visuals.ImageVisual)
#Isocurve = create_visual_node(visuals.IsocurveVisual)
#Isoline = create_visual_node(visuals.IsolineVisual)
#Isosurface = create_visual_node(visuals.IsosurfaceVisual)
#Line = create_visual_node(visuals.LineVisual)
#LinePlot = create_visual_node(visuals.LinePlotVisual)
#Markers = create_visual_node(visuals.MarkersVisual)
#Mesh = create_visual_node(visuals.MeshVisual)
#Plane = create_visual_node(visuals.PlaneVisual)
#Polygon = create_visual_node(visuals.PolygonVisual)
#Rectangle = create_visual_node(visuals.RectangleVisual)
#RegularPolygon = create_visual_node(visuals.RegularPolygonVisual)
#ScrollingLines = create_visual_node(visuals.ScrollingLinesVisual)
#Spectrogram = create_visual_node(visuals.SpectrogramVisual)
#SurfacePlot = create_visual_node(visuals.SurfacePlotVisual)
#Text = create_visual_node(visuals.TextVisual)
    

