# pylint: disable=invalid-name
"""
    Helperclass that reads the binning xml file
"""

import math
import numpy as np
import xml.etree.ElementTree as ET

class XMLHandler:

    def __init__(self, particle_name, filename='binning.xml'):

        tree = ET.parse(filename)
        root = tree.getroot()

        self.r_bins = []
        self.a_bins = []
        self.nBinAlphaPerlayer = []
        self.alphaListPerLayer = []

        self.r_edges = []
        self.r_midvalue = []
        self.r_midvalueCorrected = []
        self.relevantlayers = []
        self.layerWithBinningInAlpha = []

        self.eta_edges = []
        self.phi_edges = []
        self.eta_bins = []
        self.phi_bins = []

        self.etaRegion = 0

        found_particle = False
        for particle in root:
            if particle.attrib["name"] == particle_name:
                found_particle = True
                for layer in particle:
                    self.ReadPolarCoordinates(layer)
        if not found_particle:
            raise ValueError('Particle {} not found in {}'.format(particle_name, filename))

        self.totalBins = 0
        self.bin_number = []

        self.eta_all_layers = []
        self.phi_all_layers = []

        self.SetEtaAndPhiFromPolar()
        self.bin_edges = [0]
        for i in range(len(self.bin_number)):
            self.bin_edges.append(self.bin_number[i] + self.bin_edges[i])

    def ReadPolarCoordinates(self, subelem):
        bins = 0
        r_list = []
        str_r = subelem.attrib.get('r_edges')
        r_list = [float(s) for s in str_r.split(',')]
        bins = len(r_list) - 1

        self.r_edges.append(r_list)
        self.r_bins.append(bins)
        layer = subelem.attrib.get('id')

        bins_in_alpha = int(subelem.attrib.get('n_bin_alpha'))
        self.a_bins.append(bins_in_alpha)
        self.r_midvalue.append(self.get_midpoint(r_list))
        if bins_in_alpha > 1:
            self.layerWithBinningInAlpha.append(int(layer))

    def fill_r_a_lists(self, layer):
        no_of_rbins = self.r_bins[layer]
        list_mid_values = self.r_midvalue[layer]
        list_a_values = self.alphaListPerLayer[layer]
        r_list = []
        a_list = []
        actual_no_alpha_bins = self.nBinAlphaPerlayer[layer][0]
        for j0 in range(0, actual_no_alpha_bins):
            for i0 in range(0, no_of_rbins):
                r_list.append(list_mid_values[i0])
                a_list.append(list_a_values[i0][j0])
        return r_list, a_list

    def get_midpoint(self, arr):
        middle_points = []
        for i in range(len(arr)-1):
            middle_value = arr[i] + float((arr[i+1] - arr[i]))/2
            middle_points.append(middle_value)
        return middle_points

    def SetEtaAndPhiFromPolar(self):
        self.minAlpha = -math.pi
        self.SetNumberOfBins()

        r_all_layers = []
        alpha_all_layers = []

        for layer in range(len(self.r_bins)):
            r_list, a_list = self.fill_r_a_lists(layer)
            r_all_layers.append(r_list)
            alpha_all_layers.append(a_list)

        for layer in range(len(self.r_bins)):
            eta = r_all_layers[layer] * np.cos(alpha_all_layers[layer])
            self.eta_all_layers.append(eta)
            phi = r_all_layers[layer] * np.sin(alpha_all_layers[layer])
            self.phi_all_layers.append(phi)

    def SetNumberOfBins(self):
        for layer in range(len(self.r_bins)):
            bin_no = 0
            alphaBinList = []
            nBinAlpha = []

            bin_no = self.r_bins[layer]*self.a_bins[layer]
            centres_alpha = self.get_midpoint(np.linspace(self.minAlpha,
                                                          math.pi, self.a_bins[layer]+1))
            for _ in range(self.r_bins[layer]):
                alphaBinList.append(centres_alpha)
                nBinAlpha.append(self.a_bins[layer])

            self.totalBins += bin_no
            self.bin_number.append(bin_no)
            if self.r_bins[layer] > 0:
                self.relevantlayers.append(layer)
                self.alphaListPerLayer.append(alphaBinList)
                self.nBinAlphaPerlayer.append(nBinAlpha)
            else:
                self.alphaListPerLayer.append([0])
                self.nBinAlphaPerlayer.append([0])

    def GetTotalNumberOfBins(self):
        return self.totalBins

    def GetBinEdges(self):
        return self.bin_edges

    def GetEtaPhiAllLayers(self):
        return self.eta_all_layers, self.phi_all_layers

    def GetRelevantLayers(self):
        return self.relevantlayers

    def GetLayersWithBinningInAlpha(self):
        return self.layerWithBinningInAlpha

    def GetEtaRegion(self):
        return self.etaRegion
