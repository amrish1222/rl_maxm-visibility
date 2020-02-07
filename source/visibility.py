# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 01:33:51 2020

@author: amris
"""

import copy
from skgeom import sg

class Visibility:
    def __init__(self,length, width):
        self.boundary_obs = sg.arrangement.Arrangement()
        self.visibilityPolygon = None
        
    def addGeom2Arrangement(self, pts):
        # pts in 2D list
        edges = self.pts2Edges(pts)
        for ed in edges:
            self.boundary_obs.insert(ed)
        
    def pts2Edges(pts):
        edges = []
        for i in range(1,len(pts)):
            e = sg.Segment2(sg.Point2(pts[i-1][0],pts[i-1][1]), sg.Point2(pts[i][0],pts[i][1]))
            edges.append(e)
        e = sg.Segment2(sg.Point2(pts[len(pts)-1][0],pts[len(pts)-1][1]), sg.Point2(pts[0][0],pts[0][1]))
        edges.append(e)
        return edges
    
    def getSgPolyFromVx(self, vx):
        allEdges = []
        for e in vx.halfedges:
            edge = [e.source().point(), e.target().point()]
            edgeRev = [e.target().point(), e.source().point()]
            if not edgeRev in allEdges and not edge in allEdges:
                allEdges.append([e.source().point(), e.target().point()])
        
        vsbltyPoly = []
        prevEndPt = allEdges[0][1]
        vsbltyPoly.append( allEdges[0][0])
        allEdges.pop(0)
        
        while len(allEdges)>0:
        #    print(len(allEdges))
            for e in allEdges:
                if prevEndPt == e[0]:
                    vsbltyPoly.append(prevEndPt)
                    prevEndPt = e[1]
                    allEdges.remove(e)
                elif prevEndPt == e[1]:
                    vsbltyPoly.append(prevEndPt)
                    prevEndPt = e[0]
                    allEdges.remove(e)
                    break
        return vsbltyPoly
    
    def getVisibilityPolygon(self,fromPt):
        vs = sg.RotationalSweepVisibility(self.boundary_obs)
        q = sg.Point2(fromPt[0], fromPt[1])
        face = self.boundary_obs.find(q)
        vx = vs.compute_visibility(q, face)
        visibilityPolygon = self.getSgPolyFromVx(vx)
        self.visbilityPolygon = visibilityPolygon
        return visibilityPolygon
        
    
    def isPtinPoly(self, pt):
        # pt as a list
        position = self.visibilityPolygon.oriented_side(sg.Point2(pt[0],pt[1]))
        if position == sg.Sign.POSITIVE:
            return 1
        elif position == sg.Sign.NEGATIVE:
            return -1
        elif position == sg.Sign.ZERO:
            return 0
        