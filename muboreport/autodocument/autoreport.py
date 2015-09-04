# -*- coding: utf-8 -*-
"""
(c) 2015 Dipl. Ing. Johannes Eckstein

Created on Mon Jun 17 16:45:06 2013

-Objektorientierte Datenstrukturierung
-Automatische Bericht-Erstellung (mit reportlab)
-Intern vollständig auf Vektorgrafik basierend -> kleine, immer anzeige- und druck- optimale Dokumente!

"""

import re

#import xlrd as xr
import numpy as np

#import matplotlib
#matplotlib.use('PDF')
import matplotlib.pyplot as plt
import cStringIO
from reportlab.platypus.doctemplate import BaseDocTemplate, PageTemplate, NextPageTemplate, _doNothing
from pdfrw import PdfReader
from pdfrw.buildxobj import pagexobj
from pdfrw.toreportlab import makerl

from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

from reportlab.graphics.shapes import Drawing, Rect, String

from scipy.stats import gaussian_kde
from scipy.stats import kurtosis
from scipy.stats import variation
from scipy.stats import skew

from reportlab.lib.pagesizes import A4
#from reportlab.lib.pagesizes import landscape,portrait
from reportlab.lib.units import cm #, mm

from  reportlab.platypus.frames import Frame

from reportlab.lib import colors

from  reportlab.lib.styles import ParagraphStyle as PS
#from  reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate 

from reportlab.platypus import Image, Paragraph, PageBreak, Table, Spacer, Flowable, SimpleDocTemplate

#from reportlab.platypus.tables import LongTable

from reportlab.platypus.doctemplate import LayoutError
from reportlab.platypus.tableofcontents import TableOfContents
#from reportlab.pdfgen.pdfimages import PDFImage

from reportlab.lib.styles import getSampleStyleSheet 

#from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart, HorizontalBarChart
from reportlab.graphics import renderPM

from hashlib import sha1
from reportlab.pdfgen import canvas

from reportlab.rl_config import defaultPageSize
from reportlab.lib.units import inch
PAGE_HEIGHT=defaultPageSize[1]; PAGE_WIDTH=defaultPageSize[0]

#import file_io as fio

Title = "Hello world"
pageinfo = "platypus example"

def myFirstPage(canvas,doc):
    canvas.saveState()
    canvas.setFont('Times-Bold',16)
    canvas.drawCentredString(PAGE_WIDTH/2.0, PAGE_HEIGHT-108, Title)
    canvas.setFont('Times-Roman',9)
    canvas.drawString(inch, 0.75 * inch, "First Page / %s" % pageinfo)
    canvas.restoreState()

def myLaterPages(canvas,doc):
    canvas.saveState()
    canvas.setFont('Times-Roman',9)
    canvas.drawString(inch, 0.75 * inch, "Page %d %s" % (doc.page, pageinfo))
    canvas.restoreState()

class MyDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kw):
        self.figCount = 0
        self.allowSplitting = 0
        apply(BaseDocTemplate.__init__, (self, filename), kw)
        template = PageTemplate('normal', [Frame(2.5*cm, 2.5*cm, 15*cm, 25*cm, id='F1')])
        self.addPageTemplates(template)

    _invalidInitArgs = ('pageTemplates',)

    def handle_pageBegin(self):
        '''override base method to add a change of page template after the firstpage.
        '''
        self._handle_pageBegin()
        self._handle_nextPageTemplate('Later')

    def build(self,flowables,onFirstPage=_doNothing, onLaterPages=_doNothing, canvasmaker=canvas.Canvas):
        """build the document using the flowables.  Annotate the first page using the onFirstPage
               function and later pages using the onLaterPages function.  The onXXX pages should follow
               the signature

                  def myOnFirstPage(canvas, document):
                      # do annotations and modify the document
                      ...

               The functions can do things like draw logos, page numbers,
               footers, etcetera. They can use external variables to vary
               the look (for example providing page numbering or section names).
        """
        self._calc()    #in case we changed margins sizes etc
        frameT = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id='normal')
        self.addPageTemplates([PageTemplate(id='First',frames=frameT, onPage=onFirstPage,pagesize=self.pagesize),
                        PageTemplate(id='Later',frames=frameT, onPage=onLaterPages,pagesize=self.pagesize)])
        if onFirstPage is _doNothing and hasattr(self,'onFirstPage'):
            self.pageTemplates[0].beforeDrawPage = self.onFirstPage
        if onLaterPages is _doNothing and hasattr(self,'onLaterPages'):
            self.pageTemplates[1].beforeDrawPage = self.onLaterPages
        BaseDocTemplate.build(self,flowables, canvasmaker=canvasmaker)

    # Entries to the table of contents can be done either manually by
    # calling the addEntry method on the TableOfContents object or automatically
    # by sending a 'TOCEntry' notification in the afterFlowable method of
    # the DocTemplate you are using. The data to be passed to notify is a list
    # of three or four items countaining a level number, the entry text, the page
    # number and an optional destination key which the entry should point to.
    # This list will usually be created in a document template's method like
    # afterFlowable(), making notification calls using the notify() method
    # with appropriate data.

    def afterFlowable(self, flowable):
        "Registers TOC entries."
        if flowable.__class__.__name__ == 'Paragraph':
            text = flowable.getPlainText()
            style = flowable.style.name
            if style == 'Heading1':
                level = 0
            elif style == 'Heading2':
                level = 1
            elif style == 'Heading3':
                level = 2
            else:
                return
            E = [level, text, self.page]
            #if we have a bookmark name append that to our notify data
            bn = getattr(flowable,'_bookmarkName',None)
            if bn is not None: E.append(bn)
            self.notify('TOCEntry', tuple(E))
            if level <= 1:
                key = 'ch%s' % flowable.identity()
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text,
                                          key, level=level, closed=True)  
                                        
    def figcounter(self):
        self.figCount += 1
        return str(self.figCount)

def printl(*args):
    try:
        log.printline(args)
    except NameError:
        pass
    except AttributeError:
        print args

def printt(Table,Header):
    """
    Two arguments accepted, Table and Header
    """
    try:
        log.printtable(Table,Header)
    except NameError:
        pass
    except AttributeError:
        print "Error in call to printt"

#def GetLogFile(log=log):
#    if path_ausw == None:
#        print "path ausw must be set"
#        exit
#    if logname == None:
#        print "logname must be set"
#        exit
#    
#    if log == None:
#        log = fio.log = fio.ErrorLog(logname,path=path_ausw)
#        return  fio.log
#    else:
#        print "log file already exists"

def doHeading(ch,text,sty,parts):
    """
    function that makes headings
    """
    #create bookmarkname
    bn=sha1(str(ch)+text+sty.name).hexdigest()
    #modify paragraph text to include an anchor point with name bn
    h=Paragraph(text+'<a name="%s"/>' % bn,sty)
    #store the bookmark name on the flowable so afterFlowable can see this
    h._bookmarkName=bn
    parts.append(h)
    return parts

def doImage(Image,doc,parts,titlename):
    ###############################################################################
    #### Image ########t############################################################
    ###############################################################################
    thisImage = Image
    if not thisImage == None:
        
        factor = doc.width/thisImage.drawWidth
        thisImage.drawHeight = thisImage.drawHeight * factor
        thisImage.drawWidth  = thisImage.drawWidth  * factor
        parts.append(thisImage)
        para = Paragraph(u"Fig. " +  doc.figcounter() + u" " + titlename + u" Vertikal", caption)
        parts.append(para)
    ###############################################################################
        return parts
    else:
        return ""

# Define Styles
style = getSampleStyleSheet()

centered = PS(name = 'centered',
              fontSize = 18,
              leading = 16,
              alignment = 1,
              #spaceBefore = 2,
              spaceAfter = 20)

ht = style["Title"]

h1 = style["Heading1"]
#    PS(
#    name = 'Heading1',
#    fontSize = 14,
#    leading = 16)

h2 = style["Heading2"]
#    PS(name = 'Heading2',
#    fontSize = 12,
#    leading = 14)
    
h3 = style["Heading3"]
    
# Another Way to define own Style for caption
caption = style["Heading6"]
caption.alignment = 1
caption.spaceBefore = 2

def getTabelOfContents():
    """
    returns toc with 3 customized headings
    """
    toc = TableOfContents()
    toc.levelStyles = [
        PS(fontSize=12, name='TOCHeading1', leftIndent=20, firstLineIndent=-20, spaceBefore=6, leading=14), #fontName='Times-Bold', 
        PS(fontSize=10, name='TOCHeading2', leftIndent=40, firstLineIndent=-20, spaceBefore=4, leading=12),
        PS(fontSize=8, name='TOCHeading3', leftIndent=50, firstLineIndent=-20, spaceBefore=2, leading=10),
        ]
    return toc

class PdfImage(Flowable):
    """
    PdfImage wraps the first page from a PDF file as a Flowable
    which can be included into a ReportLab Platypus document.
    Based on the vectorpdf extension in rst2pdf (http://code.google.com/p/rst2pdf/)
    """

    def __init__(self, filename_or_object, width=None, height=None, kind='direct'):
        # If using StringIO buffer, set pointer to begining
        if hasattr(filename_or_object, 'read'):
            filename_or_object.seek(0)
        page = PdfReader(filename_or_object, decompress=False).pages[0]
        self.xobj = pagexobj(page)
        self.imageWidth = width
        self.imageHeight = height
        x1, y1, x2, y2 = self.xobj.BBox

        self._w, self._h = x2 - x1, y2 - y1
        if not self.imageWidth:
            self.imageWidth = self._w
        if not self.imageHeight:
            self.imageHeight = self._h
        self.__ratio = float(self.imageWidth)/self.imageHeight
        if kind in ['direct','absolute'] or width==None or height==None:
            self.drawWidth = width or self.imageWidth
            self.drawHeight = height or self.imageHeight
        elif kind in ['bound','proportional']:
            factor = min(float(width)/self._w,float(height)/self._h)
            self.drawWidth = self._w*factor
            self.drawHeight = self._h*factor

    def wrap(self, aW, aH):
        """
        returns draw- width and height 
        """
        return self.drawWidth, self.drawHeight

    def drawOn(self, canv, x, y, _sW=0):
        """
        translates Bounding Box and scales the given canvas 
        """
        if _sW > 0 and hasattr(self, 'hAlign'):
            a = self.hAlign
            if a in ('CENTER', 'CENTRE', TA_CENTER):
                x += 0.5*_sW
            elif a in ('RIGHT', TA_RIGHT):
                x += _sW
            elif a not in ('LEFT', TA_LEFT):
                raise ValueError("Bad hAlign value " + str(a))

        xobj = self.xobj
        xobj_name = makerl(canv._doc, xobj)

        xscale = self.drawWidth/self._w
        yscale = self.drawHeight/self._h

        x -= xobj.BBox[0] * xscale
        y -= xobj.BBox[1] * yscale

        canv.saveState()
        canv.translate(x, y)
        canv.scale(xscale, yscale)
        canv.doForm(xobj_name)
        canv.restoreState()

def barChart(daten,Versuch,Phaenomene,path=None,vMin=1,vMax=6):
    """
    Plots data to a Drawing and returns the Drawing
    """
    #Festlegen der Gesamtgröße in Pixel
    d = Drawing(500,160)
    #Daten für das Diagramm
    #daten = [(10,6,8)]
    #Anlegen des Diagramms
    diagramm = HorizontalBarChart()
    #Positionierung und Größe des Diagramms
    diagramm.x = 10
    diagramm.y = 30
    diagramm.height = 100
    diagramm.width = 400
    #Hinzufügen der Daten
    #diagramm.reversePlotOrder = 1
    diagramm.data = daten
    #Y-Achse (in ReportLab „valueAxis“) formatieren
    diagramm.valueAxis.valueMin = vMin
    diagramm.valueAxis.valueMax = vMax
    diagramm.valueAxis.valueStep = 1
    #X-Achse (in ReportLab „categoryAxis“) formatieren
    diagramm.categoryAxis.categoryNames = Phaenomene
    #Diagramm zeichnen
    d.add(diagramm)
    if not path == None:
        Versuch = path + Versuch    
    renderPM.drawToFile(d, Versuch + ".png", 'PNG')    
    #d = Paragraph(d, centered)
    d.scale(0.8,0.8)
    
    return d

def barHorizontal(daten,Versuch,Phaenomene,strds,path=None,vectorgraphics=True,show=False,vMin=1.,vMax=9,lowmid=3.5,midhigh=6.5,leftc="g",rightc="m"):
    """
    Horizontal Bar Plot with overlayed errorplot
    """   
    
    def choose_color(value):
        if ( value >= vMin and value < lowmid ):
            color=leftc
        elif ( value >= lowmid and value < midhigh ):
            color="y"
        elif ( value >= midhigh and value <= vMax ):
            color=rightc
        else:
            color="w"
        return color
    
    imgdata = cStringIO.StringIO()
    fig = plt.figure(figsize=(16, 8), dpi=60)
    
    ax = fig.add_subplot(111)
    
    ax.barh(np.arange(len (daten)), daten, height=0.8, left=0, xerr=strds,color=[choose_color(value) for value in daten])
    
    ax.minorticks_off()
    #ax.set_yticks() #np.arange(len(daten))+0.5, , size='small'
    
    ax.set_yticks(np.arange(len(daten))+0.4)
    #here we change the labels
    ax.set_yticklabels( tuple( [unicode(i) for i in Phaenomene] ),fontsize=20 )
    
    ax.set_xticklabels( tuple( [unicode(i) for i in np.arange(vMin,vMax+1,1)] ),fontsize=20 ) #-5
    
    ax.set_xlim(left=vMin, right=vMax)
    ax.set_ylim(bottom=0.0)
    
    plt.tight_layout()
    plt.savefig(imgdata,format='PDF')
    
    if show:
        plt.show()
    
    if vectorgraphics:
        return PdfImage(imgdata)    

def plotHist(data,name,subname=u"Noten der Probanden für das Phänomen",spec="der Noten-Vergabe",show=False,text=None,Versuch=None,path=None,vectorgraphics=True,N=None,Min=1,Max=6):
    """
    plotting historam, returns vector image as databuffer or Image as Portable Network Graphic
    """

    def autolabel(rects,heights):
        # attach some text labels
        #i = 0
        for i,rect in enumerate(rects):
            recheight = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., (1.05)*recheight, '%.0f'%float(heights[i]),
                    ha='center', va='bottom')    
            #i += 1
    
    imgdata = cStringIO.StringIO()
    
    fig = plt.figure(figsize=(16, 8), dpi=60)
    
    ax = fig.add_subplot(111)

     # number of Marks to be displayed
    if not N == None:
        mybins = np.arange(Min,Max+1,1)
    else:
        mybins = N
        
    print "mybins", mybins
    
    try:
        if not (len(data) < 6):
            print data
            density = gaussian_kde(data)
            density.covariance_factor = lambda : .25
            density._compute_covariance()
    
            xs = np.arange(Min,float(Max+1),1.)
    
            ys = density(xs)
    
            ax.plot(xs, ys, antialiased=True, linewidth=2, color="#A81450", label=u"Gauss-Kerndichteschätzung")
            ax.fill_between(xs, ys, alpha=.5, zorder=5, antialiased=True, color="#E01B6A")
        
    except [AttributeError,ValueError]: #np.linalg.linalg.LinAlgError,
        pass
    
    #n, bins, patches = ax.hist(data, 6, normed=1, histtype='bar', align='mid', label="Stichprobenverteilung "+ name)
    height, bins = np.histogram(data,bins=mybins)
    
    normheight = np.linalg.norm(height)
    
    print "bins", bins
    print "height", height
    print "normheight", normheight
    
    rects = ax.bar(bins[:-1], height/normheight, align='center', label="Stichprobenverteilung "+ name)
        
    autolabel(rects,height)

    #plt.setp(patches, 'facecolor', 'g', 'alpha', 0.85)
    
    ax.minorticks_off()
    
    ax.set_xlabel(subname + " " + name, fontsize=20)
    ax.set_ylabel(u"rel. Häufigkeit " + spec, fontsize=20)
    
    leg = ax.legend(loc='best', fancybox=True) #'lower left'
    leg.get_frame().set_alpha(0.5)
    lbox = leg.get_bbox_to_anchor()
    
    text_x = lbox.width/1000.
    #text_y = lbox.height/1000.
    
    fbox = ax.get_position()
    
    print bins.min(), bins.max()
    
    ax.set_xlim(left=bins.min()-0.4, right=bins.max()+0.4)
    ax.set_ylim(bottom=0.0)
    
    #frame_x = fbox.width
    frame_y = fbox.height
    
    #print frame_x,text_x, frame_y,text_y
    
    if not text == None:
        
        #text = {"key": 0.2348}
        
        val_tuple = ()
        genstr = ''
        
        for key in text.iterkeys():
            
            val_tuple = val_tuple.__add__( (text[key],) )
            genstr += key + ' %.2f' + '\n'
        
        #print genstr
        #print val_tuple
        textstr = genstr%val_tuple
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        ax.text(text_x, frame_y-0.1, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='bottom', bbox=props)
    
    if ( not path == None ) and ( not Versuch == None ):
        imgname = path + Versuch + name + ".png"
        #print imgname
    else:
        imgname = name + ".png"
    
    print imgname
    
    plt.savefig(imgname)
    #plt.savefig(imgdata,format='PDF')
    
    if show:
        plt.show()
    
    if vectorgraphics:
        try:
            return PdfImage(imgdata)
        except:
            return Image(imgname)
    else:
        return Image(imgname)
        


def getDiademImage(Vgl,fill=2,path=r"C:\UserData\eckstjo\PhenSchu\PhaenomenAbgleich\Diadem-Plots/",typ="Aufbau"):
    """
    helper Function to get pixel graphic
    """
    #PhaenSchu_Vgl_02_Stuckern.PNG
    # another way to fill with zeros: str(1).zfill(fill)
    imgname = path + "PhaenSchu_Vgl_%0*d_" % (fill, Vgl) + typ + ".PNG"
    print imgname
    return Image(imgname)

def reformatData(Data,Vgl,Phe):
    """
    Transform data to be striped along probe axis
    """
    IterData = Data.iterkeys()
    s = (len(Vgl),len(Phe),IterData.__length_hint__())
    
    NewData = np.zeros(s)
    textdata = []
    
    i=0
    
    for key in IterData:
        #print key
        #print Vgl[i]
        NewData[:,:,i], txt = Data[key]
        textdata.append(txt)
        i+=1
        
    return NewData, textdata  

class ProbandenDaten():
    def __init__(self):
        """
        initialize some attributes and an empty list
        """
        methods = ["mean","std","min","max","len"]
        for method in methods:
            setattr(self, method, None)
        setattr(self,"Event",[])
        setattr(self,"Text","")

def Convert2PdfReport(doc,parts,Daten,outpath,Fahrzeuge,Note=None,Carnames=None): #,outname,
    """
    Plots Data and Graphics to a Portable Document File
    """ 
    

    numbers = [int(re.findall(r'\d+', item)[0]) for item in Fahrzeuge]
    idxs = [numbers.index(x) for x in sorted(numbers)]
    
       
    Fahrzeuge = [Fahrzeuge[this] for this in idxs]
    print "Fahrzeuge/Vergleiche:",Fahrzeuge
    #print "Phaenomene:",Phaenomene

    if not Note == None:
        vMin=np.min(Note.Note)
        vMax=np.max(Note.Note)
        if Note.typ == "abs":
            lowmid=3.5
            midhigh=6.5
            leftc="g"
            rightc="m"
            leftct=colors.limegreen
            rightct=colors.pink
            
        elif Note.typ == "rel":
            lowmid=-1.5
            midhigh=1.5
            leftc="m"
            rightc="g"
            leftct=colors.pink
            rightct=colors.limegreen
            

    for i,Fahrzeug in enumerate(Fahrzeuge):
        print Daten.viewkeys()
        Vergleich = Daten[Fahrzeug] #,Meinungen
        title = u"Auswertung für " + str(Fahrzeug)
        if not Carnames == None:
            htitle = title + " " + str(Carnames[Fahrzeug])
        
        Means = []#()
        Stds = []#()
        Data = []
          
        celldata = [["" for k in range(7+2)] for m in range(7+1)]    
        
        #celldata[0:-1][0] = [u"Heben",u"Nicken",u"Wanken",u"Werfen",u"(Mikro-) Stuckern",u"Stößigkeit"] #Phe
        #celldata[0][0:-1] = ["","Mittelwert","Standardabweichung","Minimum","Maximum","Stichprobenmenge"]
        
        parts = doHeading(Fahrzeuge[i],htitle, h1, parts)
     
        parts = doHeading(Fahrzeuge[i],u"Fahrzeug Übersicht", h2,parts)
        Phaenomene = []
        for j,key  in enumerate(Vergleich.iterkeys()):
            Phaenomene.append(key)
            Phaenomen = Vergleich[key]
            #Means = Means.__add__( (Phaenomen.mean,) )
           # Stds = Stds.__add__( (Phaenomen.std,) )
            Means.append(Phaenomen.mean)
            Stds.append(Phaenomen.std/2.)
            Data.append(Phaenomen)
            
            try:
                #print Phe[j]
                #print celldata
                try:
                    celldata[j+1][0] = unicode(Phaenomene[j])
                except IndexError:
                    print "Error:"
                    #print celldata, Phaenomene[j]
                
                if not Phaenomen.len == 0:
                    try:
                        celldata[0][1] = "Mittelwert"
                        celldata[j+1][1] = '%1.3f' % (Phaenomen.mean)
                        celldata[0][2] = "Standardabweichung"
                        celldata[j+1][2] = '%1.3f' % (Phaenomen.std)
                        celldata[0][3] = "Minimum"
                        celldata[j+1][3] = Phaenomen.min
                        celldata[0][4] = "Maximum"
                        celldata[j+1][4] = Phaenomen.max
                        celldata[0][5] = "Stichprobenmenge"
                        celldata[j+1][5] = Phaenomen.len
                    except:
                        pass
                    
                else:
                    
                    para = Paragraph(u"Zu "+unicode(Phaenomene[j])+u": Keine Auswertung Möglich,", style["Normal"])
                    parts.append(para)
                    
                    para = Paragraph("Anzahl Vergebener Noten:" + str(Phaenomen.len), style["Normal"])
                    parts.append(para)                
                
            except LayoutError:
                print "Layout error detected, could not create the Document Template"
                
        #thisDrawing = barChart(Means,title+"Mittelwerte",Phaenomene,path=outpath,vMin=-4,vMax=4)
                
        thisDrawing = barHorizontal(Means[::-1],title+"Mittelwerte",Phaenomene[::-1],Stds[::-1],path=outpath,vMin=vMin,vMax=vMax,lowmid=lowmid,midhigh=midhigh,leftc=leftc,rightc=rightc) # relative: 
        
        factor = (doc.width*0.85)/thisDrawing.drawWidth
        thisDrawing.drawHeight = thisDrawing.drawHeight * factor
        thisDrawing.drawWidth  = thisDrawing.drawWidth  * factor
        
        parts.append(thisDrawing)
        
        para = Paragraph(u"Mittelwerte der Phänomene mit Standardabweichung", caption)
        parts.append(para)
        
        parts.append(Spacer(1, 12))
        
        mystyle=[ 
                    ('LINEABOVE',(0,0),(-1,0),1,colors.blue),
                    ('LINEABOVE',(0,1),(-1,1),1,colors.blue),
                    ('LINEBEFORE',(1,1),(1,-1),1,colors.pink),
                    ('LINEBELOW',(0,-1),(-1,-1),1,colors.blue),]
            
        for l,key  in enumerate(Vergleich.iterkeys()):
            value = Vergleich[key].mean
            if ( value >= vMin and value < lowmid ):
                mystyle.append(('BACKGROUND',(1,l+1),(1,l+1),leftct))
            elif ( value >= lowmid and value < midhigh ):
                mystyle.append(('BACKGROUND',(1,l+1),(1,l+1),colors.khaki))
            elif ( value >= midhigh and value <= vMax ):
                mystyle.append(('BACKGROUND',(1,l+1),(1,l+1), rightct))
            else:
                pass
        
        t=Table(celldata, style=mystyle)
        #colors.brown
        parts.append(t)
        
        parts.append(Spacer(1, 12))
        parts.append(PageBreak())
        parts = doHeading(Fahrzeuge[i],u"Histogramme der Phänomene", h2,parts)
        
        for m,data in enumerate(Data):
            if not data.len == 0:
                
                text = {}
                
                text.update({"Standardabweichung":data.std})
                text.update({"Varianz":variation(data.Event)})
                text.update({"Schiefe":skew(data.Event)})
                text.update({"Kurtosis":kurtosis(data.Event)})
                
                thisImage = plotHist(data.Event,Phaenomene[m],show=False,text=text,Versuch=title,path=outpath,N=Note.Note,Min=vMin,Max=vMax)
                #except:
                #    continue

                factor = (doc.width*0.85)/thisImage.drawWidth
                thisImage.drawHeight = thisImage.drawHeight * factor
                thisImage.drawWidth  = thisImage.drawWidth  * factor
                
                parts = doHeading(Fahrzeuge[i],u"Phänomen " + unicode(Phaenomene[m]), h3,parts)
                #para = Paragraph(u"Phänomen " + str(Phe[idxs[m]]), style["Heading3"])
                #parts.append(para)
                
                parts.append(thisImage)
        parts.append(PageBreak())
        parts = doHeading(Fahrzeuge[i],u"Verbale Bemerkungen", h2,parts)
        
        for o,Phaenomen in enumerate(Phaenomene):
        
            if not len(Vergleich[Phaenomen].Text) == 0:
                parts = doHeading(Fahrzeuge[i],u"Probandenmeinung " + unicode(Phaenomen) , h3,parts)
                #print Phaenomene[o], Meinungen[o]
                para = Paragraph(Vergleich[Phaenomen].Text, style["Normal"])
                parts.append(para)         
        
        parts.append(PageBreak())
        
    plt.close('all')        
        
    try:
        
        return parts
        #doc.build(parts)
        #doc.multiBuild(parts)

    except LayoutError:
        return LayoutError("there is an error with the Layout")
    
def Auswertung(NewData,textdata,Vgl,Phe):
    """
    actual Post processing main Function
    """
    nvgl, nphe, prob = NewData.shape
    
    Vergleiche = {}
    print "===================================="
    for Vergleich in Vgl:

        k = Vgl.index(Vergleich)
        print "Vergleich", Vergleich
        print "===================================="

        text = ""
        
        for l in range(prob):
            txt = textdata[l][k]
            #print textdata
            if not (txt == None or txt == ''):
                text += " | " + txt
            else:
                pass

        Phaenomene = {}

        for Phaenomen in Phe:
            j = Phe.index(Phaenomen)
            print u"Phänomen", Phaenomen
            data = ProbandenDaten()
            
            for l in range(prob):
                value = NewData[k,j,l]
                if not value == 0. :
                    data.Event.append(value)                

            #print "===================================="
            if not len(data.Event) == 0:
                data.mean = np.mean(data.Event)
                data.std = np.std(data.Event)
                data.min = np.min(data.Event)
                data.max = np.max(data.Event)
                
                #print "Mittelwert:", data.mean
                #print "Standardabweichung:", data.std
                #print "Minimum:", data.min
                #print "Maximum:", data.max
            else:
                u"Keine Auswertung Möglich"
            data.len = len(data.Event)
            print "Anzahl Vergebener Noten:", data.len
            print "===================================="
            Phaenomene.update({Phaenomen:data})
        #print "===================================="
        Vergleiche.update({Vergleich:(Phaenomene,text)})
    return Vergleiche

#def Statistik(NewData,TextData,Phaenomene,Noten,Fahrzeuge):
#    """
#    actual Post processing main Function
#    """
#    nphen,nnote,nprob,nfzg = NewData.shape
#    
#    print "nphen",nphen,"nnote",nnote,"nprob",nprob,"nfzg",nfzg
#    
#    #print TextData.shape
#    
#    #NewData:
#    #(Phaenomene,Noten,Probanden,Fahrzeuge)
#    # über die Probanden muss jeweils der Mittelwert gebildet werden!!!
#
#    Fzg = {}
#    for j in range(nfzg):
#        Phen = {}
#        
#        text = ["" for l in range(nphen)]
#        
#        for m in range(nphen):
#            print Phaenomene[m]
#            data = ProbandenDaten()
#            textbuff = ""
#            for i in range(nprob):
#                
#                if not TextData[i][j][m] == u'':
#                    print "nprob",i, "nfzg",j, "nphen",m
#                    #print "TextData", TextData[i][j][m]
#                    textbuff += unicode(TextData[i][j][m]) + " / "
#                for n in range(nnote):
#                    if NewData[m,n,i,j]:
#                        value = Noten[n]
#                        #print Fahrzeuge[j]
#                        #print Phaenomene[m], value
#                        data.Event.append(value)
#            try:
#                if not len(data.Event) == 0:
#                    data.mean = np.mean(data.Event)
#                    data.std = np.std(data.Event)
#                    data.min = np.min(data.Event)
#                    data.max = np.max(data.Event)
#                    data.Text = textbuff[:-3]
#                    
#                    #print "Mittelwert:", data.mean
#                    #print "Standardabweichung:", data.std
#                    #print "Minimum:", data.min
#                    #print "Maximum:", data.max
#                else:
#                    u"Keine Auswertung Möglich"
#                data.len = len(data.Event)
#            except:
#                pass
#            Phen.update({Phaenomene[m]:data})
#            #print textbuff
#            text[m] = textbuff[:-3]
#        Fzg.update({Fahrzeuge[j]:(Phen,text)})
#                        
#    return Fzg

#def Convert2PdfTables(doc,parts,Vergleiche,Vgl,Phe,outpath): #,outname,
#    """
#    Plots Data and Graphics to a Portable Document File
#    """      
#    #doc = SimpleDocTemplate(outname,pagesize=A4)  # pagesize=lsize,rightMargin=0,leftMargin=0,topMargin=0,bottomMargin=0
#    #parts = []
#    
#    #toc = TableOfContents() 
#    #parts.append(toc)
#
#    for i in range(len( Vgl )):
#        #print Vgl[i]
#        Vergleich,Meinungen = Vergleiche[Vgl[i]]
#        title = u"Vergleich " + str(Vgl[i])
#        Means = ()
#        Data = []
#        celldata = [["" for k in range(len( Phe )+2)] for m in range(7)]       
#        
#        #celldata[0:-1][0] = [u"Heben",u"Nicken",u"Wanken",u"Werfen",u"(Mikro-) Stuckern",u"Stößigkeit"] #Phe
#        #celldata[0][0:-1] = ["","Mittelwert","Standardabweichung","Minimum","Maximum","Stichprobenmenge"]
#        
#        parts = doHeading(Vgl[i],title, h1, parts)
#        #print parts
#        #para = Paragraph(, style["Heading1"])
#        #parts.append(para)
#        
#        parts = doHeading(Vgl[i],u"Objektive Kennwerte", h2,parts)
#        
#        #para = Paragraph(u"Objektive Kennwerte", style["Heading2"])
#        #parts.append(para)
#        
#        #Aufbau
#        thisImage = getDiademImage(Vgl[i])
#        
#        #factor = doc.width/thisImage.drawWidth
#        factor = (doc.height/2.5)/thisImage.drawHeight
#        
#        print thisImage.drawHeight
#        
#        thisImage.drawHeight = thisImage.drawHeight * factor
#        thisImage.drawWidth  = thisImage.drawWidth  * factor
#        
#        parts.append(thisImage)
#        
#        #Stuckern
#        thisImage = getDiademImage(Vgl[i],typ="Stuckern")
#        
#        #factor = (doc.width*0.8)/thisImage.drawWidth
#        factor = (doc.height/2.5)/thisImage.drawHeight
#        
#        thisImage.drawHeight = thisImage.drawHeight * factor
#        thisImage.drawWidth  = thisImage.drawWidth  * factor
#        
#        parts.append(thisImage)
#        
#        
#        ### Subjektiv!!                                                     ###
#        
#        parts.append(PageBreak())
#        
#        parts = doHeading(Vgl[i],u"Subjektive Kennwerte", h2,parts)
#        
#        #para = Paragraph(u"Subjektive Kennwerte", style["Heading2"])
#        #parts.append(para)
#        
#        for j in range(len(Phe)):
#            
#            Phaenomen = Vergleich[Phe[j]]
#            Means = Means.__add__( (Phaenomen.mean,) )
#            Data.append(Phaenomen.Event)
#            
#            try:
#                #print Phe[j]
#                celldata[j+1][0] = str(Phe[j])
#                
#                if not Phaenomen.len == 0:
#                    celldata[0][1] = "Mittelwert"
#                    celldata[j+1][1] = '%1.3f' % (Phaenomen.mean)
#                    celldata[0][2] = "Standardabweichung"
#                    celldata[j+1][2] = '%1.3f' % (Phaenomen.std)
#                    celldata[0][3] = "Minimum"
#                    celldata[j+1][3] = Phaenomen.min
#                    celldata[0][4] = "Maximum"
#                    celldata[j+1][4] = Phaenomen.max
#                    celldata[0][5] = "Stichprobenmenge"
#                    celldata[j+1][5] = Phaenomen.len
#                    
#                else:
#                    
#                    para = Paragraph(u"Zu "+Phe[j]+u": Keine Auswertung Möglich,", style["Normal"])
#                    parts.append(para)
#                    
#                    para = Paragraph("Anzahl Vergebener Noten:" + str(Phaenomen.len), style["Normal"])
#                    parts.append(para)                
#                
#            except LayoutError:
#                print "Layout error detected, could not create the Document Template"
#                
#        thisDrawing = barChart([Means[::-1]], title + "Mittelwerte",Phe,path=outpath)
#                
#        parts.append(thisDrawing)
#        
#        para = Paragraph(u"Mittelwerte der Phänomene", caption)
#        parts.append(para)
#        
#        parts.append(Spacer(1, 12))
#        dbuf = np.asarray(Means)
#
#        # find the strogest phenomena
#        idxs = np.argsort(dbuf)
#        
#        #print idxs
#        
#        Positions = []
#        NumPlots = 3
#        if dbuf[idxs[0]] == None:
#            Init = 1
#            End = Init+NumPlots
#        else:
#            Init = 0
#            End = Init+NumPlots
#        
#        #print range(Init,End,1)
#        
#        for l in range(Init,End,1):
#            Positions.append( (1,idxs[l]+1))
#        
#        t=Table(celldata, style=[ 
#                    ('LINEABOVE',(0,0),(-1,0),1,colors.blue),
#                    ('LINEABOVE',(0,1),(-1,1),1,colors.blue),
#                    ('LINEBEFORE',(1,1),(1,-1),1,colors.pink),
#                    ('LINEBELOW',(0,-1),(-1,-1),1,colors.blue),
#                    
#                    ('BACKGROUND',Positions[0],Positions[0],colors.limegreen),
#                    ('BACKGROUND',Positions[1],Positions[1],colors.khaki),
#                    ('BACKGROUND',Positions[2],Positions[2], colors.pink),
#                    ])
#        
#        parts.append(t)
#        
#        for m in range(Init,End,1):
#            data = Data[idxs[m]]
#            if not len(data) == 0:
#                
#                text = {}
#                
#                text.update({"Standardabweichung":Vergleich[Phe[idxs[m]]].std})
#                text.update({"Varianz":variation(Vergleich[Phe[idxs[m]]].Event)})
#                text.update({"Schiefe":skew(Vergleich[Phe[idxs[m]]].Event)})
#                text.update({"Kurtosis":kurtosis(Vergleich[Phe[idxs[m]]].Event)})
#                
#                thisImage = plotHist(data,Phe[idxs[m]],show=False,text=text,Versuch=title,path=outpath,N=6)
#
#                factor = doc.width/thisImage.drawWidth
#                thisImage.drawHeight = thisImage.drawHeight * factor
#                thisImage.drawWidth  = thisImage.drawWidth  * factor
#                
#                if (m == Init+1): #(not i == 0) and
#                    pass
#                    #parts.append(PageBreak())
#                
#                parts = doHeading(Vgl[i],u"Phänomen " + str(Phe[idxs[m]]), h3,parts)
#                #para = Paragraph(u"Phänomen " + str(Phe[idxs[m]]), style["Heading3"])
#                #parts.append(para)
#                
#                parts.append(thisImage)
#        
#        parts.append(Spacer(1, 12))
#        
#        parts = doHeading(Vgl[i],u"Probandenmeinungen:", h3,parts)
#        
#        para = Paragraph(Meinungen, style["Normal"])
#        parts.append(para)         
#        
#        parts.append(PageBreak())
#        
#    try:
#        return parts
#        #doc.build(parts)
#        #doc.multiBuild(parts)
#
#    except LayoutError:
#        return LayoutError("there is an error with the Layout")

###############################################################################

if __name__ == "__main__":
    # create some Data first
    Content = {}
    Context = {}
    Chapters = [u"Kapitel 1",]
    Subchapters = [u"Unterkapitel 1", u"Unterkapitel 2"]
    
    mu, sigma = 100, 15
    #x = mu + sigma*np.random.randn(10000)
    x = np.random.uniform(1, 6, size=150)

    
    Content.update({"my first data":x})
    Context.update({"my first data":"What I always wanted to say about data that has some data title"})

    # Begin of Documentation to Potable Document
    doc = MyDocTemplate("MinimalExample.pdf")
    parts = []
    #doc.multiBuild(story)
    
    # add title
    para = Paragraph(u"Minimal Example Title", ht)
    parts.append(para)
    parts.append(PageBreak())

    # Create an instance of TableOfContents. Override the level styles (optional)
    # and add the object to the story
    toc = getTabelOfContents()
    parts.append(Paragraph('<b>Inhaltsverzeichnis</b>', centered))
    parts.append(toc)
    parts.append(PageBreak())
    
    # Begin of First Chapter
    parts = doHeading(1,Chapters[0], h1, parts)
    
    para = Paragraph(u"My Text that I can write here or take it from somewhere like shown in the next paragraph.", style["Normal"])
    parts.append(para)
    
    parts = doHeading(1,Subchapters[1], h2,parts)
    
    title = u"my first data"
    
    para = Paragraph(Context[title], style["Normal"])
    parts.append(para)

    text = {}
    
    text.update({"Standardabweichung":np.std(x)})
    text.update({"Varianz":variation(x)})
    text.update({"Schiefe":skew(x)})
    text.update({"Kurtosis":kurtosis(x)})
    
    print Content[title]
    
    thisImage = plotHist(Content[title],title,subname="",spec="",show=False,text=text,Versuch=Chapters[0],path="",N=6)

    factor = doc.width/thisImage.drawWidth
    thisImage.drawHeight = thisImage.drawHeight * factor
    thisImage.drawWidth  = thisImage.drawWidth  * factor
    
    parts.append(thisImage)
    
    para = Paragraph(u"Fig. " + str(doc.figCount) + title, caption)
    parts.append(para)
    
    #write the buffer to the document
    doc.build(parts,onFirstPage=myFirstPage,onLaterPages=myLaterPages)
    
    #doc.multiBuild(parts,onFirstPage=myFirstPage,onLaterPages=myLaterPages)
