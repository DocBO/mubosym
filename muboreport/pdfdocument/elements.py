# -*- coding: utf-8 -*-

import os

from django.conf import settings

from pdfdocument.document import cm, mm


def create_stationery_fn(*fns):
    def _fn(canvas, document):
        for fn in fns:
            fn(canvas, document.PDFDocument)
    return _fn


class ExampleStationery(object):
    def __call__(self, canvas, pdfdocument):
        left_offset = 28.6*mm

        canvas.saveState()
        canvas.setFont('%s' % pdfdocument.style.fontName, 12)
        canvas.drawString(26*mm, 284*mm, 'EuroSciPy Registration Platform')
        canvas.setFont('%s' % pdfdocument.style.fontName, 12)
        canvas.drawString(26*mm, 276*mm, u'Enthought Ltd')
        canvas.setFont('%s' % pdfdocument.style.fontName, 10)
        canvas.drawString(26*mm, 270*mm, u'The Broers Building, 3rd Floor')
        canvas.setFont('%s' % pdfdocument.style.fontName, 10)
        canvas.drawString(26*mm, 266*mm, u'Hauser Forum')
        canvas.setFont('%s' % pdfdocument.style.fontName, 10)
        canvas.drawString(26*mm, 262*mm, u'21 J.J. Thomson Avenue')
        canvas.setFont('%s' % pdfdocument.style.fontName, 10)
        canvas.drawString(26*mm, 258*mm, u'Cambridge, UK')
        canvas.setFont('%s' % pdfdocument.style.fontName, 10)
        canvas.drawString(26*mm, 254*mm, u'CB3 0FA')
        canvas.setFont('%s' % pdfdocument.style.fontName, 12)
        canvas.drawString(26*mm, 246*mm, u'EuroSciPy 2015 - The 8th European Conference on Python in Science')
        pdfdocument.draw_watermark(canvas)
        canvas.restoreState()

        canvas.saveState()
        canvas.setFont('%s' % pdfdocument.style.fontName, 6)
        for i, text in enumerate(reversed([pdfdocument.doc.page_index_string()])):
            canvas.drawRightString(190*mm, (8+3*i)*mm, text)

        logo = getattr(settings, 'PDF_LOGO_SETTINGS', None)
        if logo:
            canvas.drawImage(os.path.join(settings.APP_BASEDIR, 'metronom', 'reporting', 'images', logo[0]),
                **logo[1])

        canvas.restoreState()


class PageFnWrapper(object):
    """
    Wrap an old-style page setup function
    """

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, canvas, pdfdocument):
        self.fn(canvas, pdfdocument.doc)
