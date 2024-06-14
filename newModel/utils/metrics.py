import numpy
def binary_dc(result, reference):
    r"""
    Dice coefficient
    
    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in two images.
    
    The metric is defined as
    
    .. math::
        
        DC=\frac{2|A\cap B|}{|A|+|B|}
        
    , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).
    
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    
    Returns
    -------
    dc : float
        The Dice coefficient between the object(s) in ```result``` and the
        object(s) in ```reference```. It ranges from 0 (no overlap) to 1 (perfect overlap).
        
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    result = numpy.atleast_1d(result.astype(bool))
    reference = numpy.atleast_1d(reference.astype(bool))
    
    intersection = numpy.count_nonzero(result & reference)
    
    size_i1 = numpy.count_nonzero(result)
    size_i2 = numpy.count_nonzero(reference)
    
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0
    
    return dc

def binary_jc(result, reference):
    """
    Jaccard coefficient
    
    Computes the Jaccard coefficient between the binary objects in two images.
    
    Parameters
    ----------
    result: array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
    reference: array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.

    Returns
    -------
    jc: float
        The Jaccard coefficient between the object(s) in `result` and the
        object(s) in `reference`. It ranges from 0 (no overlap) to 1 (perfect overlap).
    
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    result = numpy.atleast_1d(result.astype(bool))
    reference = numpy.atleast_1d(reference.astype(bool))
    
    intersection = numpy.count_nonzero(result & reference)
    union = numpy.count_nonzero(result | reference)
    
    jc = float(intersection) / float(union)
    
    return jc

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if (gt.sum()>0) :
        if (pred.sum()>0) : # Algun pixel verdadero y algun pixel predicho
            iou  = binary_jc(pred, gt)   #metric.binary.jc(pred, gt)
            dice = binary_dc(pred, gt)   #metric.binary.dc(pred, gt)
            #hd   = bmetric.binary_hd(pred, gt)   #metric.binary.hd(pred, gt)
            #hd95 = bmetric.binary_hd95(pred, gt) #metric.binary.hd95(pred, gt)
            return iou, dice, 1
        else:
            return 0, 0, 1 # Algun pixel verdadero y ningun pixel predicho --> Como incorporar hd y hd95 en este caso
    else:
        if (pred.sum()>0) :
            return 0, 0, 0 # Ningun pixel verdadero y algun pixel predicho 
        else:
            return 0, 0, 0 # Ningun pixel verdadero y ningun pixel predicho  