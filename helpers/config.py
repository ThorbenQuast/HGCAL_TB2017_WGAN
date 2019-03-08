GlobalNoiseThreshold = 2.  #MIPS
SignalSmearing = 0.       
LogScale = False
maskDeadChannels = True



deadChannelList = []
for layer in range(7):
    deadChannelList.append([0,0,layer])
    deadChannelList.append([0,1,layer])
    deadChannelList.append([0,2,layer])
    deadChannelList.append([0,3,layer])
    deadChannelList.append([0,4,layer])
    deadChannelList.append([0,10,layer])
    deadChannelList.append([0,11,layer])
    deadChannelList.append([0,12,layer])
    deadChannelList.append([0,13,layer])
    deadChannelList.append([0,14,layer])
    deadChannelList.append([1,0,layer])
    deadChannelList.append([1,1,layer])
    deadChannelList.append([1,2,layer])
    deadChannelList.append([1,12,layer])
    deadChannelList.append([1,13,layer])
    deadChannelList.append([1,14,layer])
    deadChannelList.append([2,0,layer])
    deadChannelList.append([2,1,layer])
    deadChannelList.append([2,13,layer])
    deadChannelList.append([2,14,layer])
    for j in range(3,8):
        deadChannelList.append([j,0,layer])
        if j==5:
            if layer==5:
                deadChannelList.append([j,3,layer])
                deadChannelList.append([j,9,layer])
            else:
                deadChannelList.append([j,5,layer])
                deadChannelList.append([j,11,layer])            
        deadChannelList.append([j,14,layer])
    deadChannelList.append([8,0,layer])
    deadChannelList.append([8,1,layer])
    deadChannelList.append([8,13,layer])
    deadChannelList.append([8,14,layer])
    deadChannelList.append([9,0,layer])
    deadChannelList.append([9,1,layer])
    deadChannelList.append([9,13,layer])
    deadChannelList.append([9,14,layer])
    deadChannelList.append([10,0,layer])
    deadChannelList.append([10,1,layer])
    deadChannelList.append([10,2,layer])
    deadChannelList.append([10,3,layer])
    deadChannelList.append([10,11,layer])
    deadChannelList.append([10,12,layer])
    deadChannelList.append([10,13,layer])
    deadChannelList.append([10,14,layer])
    deadChannelList.append([11,0,layer])
    deadChannelList.append([11,1,layer])
    deadChannelList.append([11,2,layer])
    deadChannelList.append([11,3,layer])
    deadChannelList.append([11,4,layer])
    deadChannelList.append([11,5,layer])
    deadChannelList.append([11,7,layer])
    deadChannelList.append([11,9,layer])
    deadChannelList.append([11,10,layer])
    deadChannelList.append([11,11,layer])
    deadChannelList.append([11,12,layer])
    deadChannelList.append([11,13,layer])
    deadChannelList.append([11,14,layer])