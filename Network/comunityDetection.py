from networkx.algorithms.community import girvan_newman
from Network.FastNewman import *
class CommunityDetection:
    def __init__(self, detection_algorithm):
        self.detection_algorithm = None
        if detection_algorithm == "fastNewman":
            self.detection_algorithm = "fastNewman"
        elif detection_algorithm == "girvanNewman":
            self.detection_algorithm = "girvanNewman"

    def detection(self, network, nCommunities=2):
        if self.detection_algorithm == "fastNewman":
            return fastNewmanAlgorithm(network, nCommunities=nCommunities)
        elif self.detection_algorithm == "girvanNewman":
            return girvan_newman(network)



