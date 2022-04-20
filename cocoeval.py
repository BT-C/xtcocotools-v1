import numpy as np
import datetime
import time
from collections import defaultdict
from . import mask as maskUtils
import copy
import sys
import warnings


class NullWriter(object):

    def write(self, arg):
        pass

    def flush(self):
        pass

class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # The original Microsoft COCO Toolbox is written
    # # by Piotr Dollar and Tsung-Yi Lin, 2014.
    # # Licensed under the Simplified BSD License [see bsd.txt]
    ######################################################################
    # Updated and renamed to Extended COCO Toolbox (xtcocotool) \
    # by Sheng Jin & Can Wang in 2020. The Extended COCO Toolbox is
    # developed to support multiple pose-related datasets, including COCO,
    # CrowdPose and so on.

    def __init__(self, cocoGt=None, cocoDt=None, iouType='keypoints', sigmas=None, use_area=True):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :param iouType: 'segm', 'bbox' or 'keypoints', 'keypoints_crowd'
        :param sigmas: keypoint labelling sigmas.
        :param use_area (bool): If gt annotations (eg. CrowdPose, AIC)
                                do not have 'area', please set use_area=False.
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType keypoints')
        if sigmas is not None:
            self.sigmas = sigmas
        else:
            # The default sigmas are used for COCO dataset.
            self.sigmas = np.array(
                [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())
            self.anno_file = cocoGt.anno_file
        self.use_area = use_area

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag and score key
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if 'keypoints' in p.iouType:
                if p.iouType == 'keypoints_wholebody':
                    body_gt = gt['keypoints']
                    foot_gt = gt['foot_kpts']
                    face_gt = gt['face_kpts']
                    lefthand_gt = gt['lefthand_kpts']
                    righthand_gt = gt['righthand_kpts']
                    wholebody_gt = body_gt + foot_gt + face_gt + lefthand_gt + righthand_gt
                    g = np.array(wholebody_gt)
                    k = np.count_nonzero(g[2::3] > 0)
                    self.score_key = 'wholebody_score'
                elif p.iouType == 'keypoints_foot':
                    g = np.array(gt['foot_kpts'])
                    k = np.count_nonzero(g[2::3] > 0)
                    self.score_key = 'foot_score'
                elif p.iouType == 'keypoints_face':
                    g = np.array(gt['face_kpts'])
                    k = np.count_nonzero(g[2::3] > 0)
                    self.score_key = 'face_score'
                elif p.iouType == 'keypoints_lefthand':
                    g = np.array(gt['lefthand_kpts'])
                    k = np.count_nonzero(g[2::3] > 0)
                    self.score_key = 'lefthand_score'
                elif p.iouType == 'keypoints_righthand':
                    g = np.array(gt['righthand_kpts'])
                    k = np.count_nonzero(g[2::3] > 0)
                    self.score_key = 'righthand_score'
                elif p.iouType == 'keypoints_crowd':
                    # 'num_keypoints' in CrowdPose dataset only counts
                    # the visible joints (vis = 2)
                    k = gt['num_keypoints']
                    self.score_key = 'score'
                else:
                    g = np.array(gt['keypoints'])
                    k = np.count_nonzero(g[2::3] > 0)
                    self.score_key = 'score'

                gt['ignore'] = (k == 0) or gt['ignore']
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)

        flag_no_part_score = False
        for dt in dts:
            # ignore all-zero keypoints and check part score
            if 'keypoints' in p.iouType:
                if p.iouType == 'keypoints_wholebody':
                    body_dt = dt['keypoints']
                    foot_dt = dt['foot_kpts']
                    face_dt = dt['face_kpts']
                    lefthand_dt = dt['lefthand_kpts']
                    righthand_dt = dt['righthand_kpts']
                    wholebody_dt = body_dt + foot_dt + face_dt + lefthand_dt + righthand_dt
                    d = np.array(wholebody_dt)
                    k = np.count_nonzero(d[2::3] > 0)
                    if self.score_key not in dt:
                        dt[self.score_key] = dt['score']
                        flag_no_part_score = True
                elif p.iouType == 'keypoints_foot':
                    d = np.array(dt['foot_kpts'])
                    k = np.count_nonzero(d[2::3] > 0)
                    if self.score_key not in dt:
                        dt[self.score_key] = dt['score']
                        flag_no_part_score = True
                elif p.iouType == 'keypoints_face':
                    d = np.array(dt['face_kpts'])
                    k = np.count_nonzero(d[2::3] > 0)
                    if self.score_key not in dt:
                        dt[self.score_key] = dt['score']
                        flag_no_part_score = True
                elif p.iouType == 'keypoints_lefthand':
                    d = np.array(dt['lefthand_kpts'])
                    k = np.count_nonzero(d[2::3] > 0)
                    if self.score_key not in dt:
                        dt[self.score_key] = dt['score']
                        flag_no_part_score = True
                elif p.iouType == 'keypoints_righthand':
                    d = np.array(dt['righthand_kpts'])
                    k = np.count_nonzero(d[2::3] > 0)
                    if self.score_key not in dt:
                        dt[self.score_key] = dt['score']
                        flag_no_part_score = True
                else:
                    d = np.array(dt['keypoints'])
                    k = np.count_nonzero(d[2::3] > 0)
                if k == 0:
                    continue
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        if flag_no_part_score:
            warnings.warn("'{}' not found, use 'score' instead.".format(self.score_key))
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif 'keypoints' in p.iouType:
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def computeIoU(self, imgId, catId):
        """
        Returns ious - [D x G] array of IoU values for all pairs of detections and gt instances.
        Where D is the number of detections and G is the number of gt intances.
        Detections are sortred from the highest to lowest score before computing `ious`.
        So rows in `ious` are ordered according to detection scores.
        """
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d[self.score_key] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
        return ious

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d[self.score_key] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = self.sigmas
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            if p.iouType == 'keypoints_wholebody':
                body_gt = gt['keypoints']
                foot_gt = gt['foot_kpts']
                face_gt = gt['face_kpts']
                lefthand_gt = gt['lefthand_kpts']
                righthand_gt = gt['righthand_kpts']
                wholebody_gt = body_gt + foot_gt + face_gt + lefthand_gt + righthand_gt
                g = np.array(wholebody_gt)
            elif p.iouType == 'keypoints_foot':
                g = np.array(gt['foot_kpts'])
            elif p.iouType == 'keypoints_face':
                g = np.array(gt['face_kpts'])
            elif p.iouType == 'keypoints_lefthand':
                g = np.array(gt['lefthand_kpts'])
            elif p.iouType == 'keypoints_righthand':
                g = np.array(gt['righthand_kpts'])
            else:
                g = np.array(gt['keypoints'])

            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                if p.iouType == 'keypoints_wholebody':
                    body_dt = dt['keypoints']
                    foot_dt = dt['foot_kpts']
                    face_dt = dt['face_kpts']
                    lefthand_dt = dt['lefthand_kpts']
                    righthand_dt = dt['righthand_kpts']
                    wholebody_dt = body_dt + foot_dt + face_dt + lefthand_dt + righthand_dt
                    d = np.array(wholebody_dt)
                elif p.iouType == 'keypoints_foot':
                    d = np.array(dt['foot_kpts'])
                elif p.iouType == 'keypoints_face':
                    d = np.array(dt['face_kpts'])
                elif p.iouType == 'keypoints_lefthand':
                    d = np.array(dt['lefthand_kpts'])
                elif p.iouType == 'keypoints_righthand':
                    d = np.array(dt['righthand_kpts'])
                else:
                    d = np.array(dt['keypoints'])

                xd = d[0::3]; yd = d[1::3]
                if k1>0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)

                if self.use_area:
                    e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
                else:
                    tmparea = gt['bbox'][3] * gt['bbox'][2] * 0.53
                    e = (dx**2 + dy**2) / vars / (tmparea+np.spacing(1)) / 2

                if k1 > 0:
                    e=e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if 'area' not in g or not self.use_area:
                tmp_area = g['bbox'][2] * g['bbox'][3] * 0.53
            else:
                tmp_area =g['area']
            if g['ignore'] or (tmp_area < aRng[0] or tmp_area > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d[self.score_key] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        # https://github.com/cocodataset/cocoapi/pull/332/
        gtm  = np.ones((T,G)) * -1
        dtm  = np.ones((T,D)) * -1
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if len(ious):
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>=0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        # since all the rest of g's are ignored as well because of the prior sorting
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:

                        #-------------------------------------------------------------------
                        ''' add fake result '''
                        # assert len(dt) == len(gt)
                        # if 5 <= tind <= 7:
                        #     dtIg[tind,dind] = gtIg[0]
                        #     dtm[tind,dind] = gt[0]['id']
                        #     gtm[tind,0]     = d['id']
                        #-------------------------------------------------------------------

                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d[self.score_key] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        continue
                    # https://github.com/cocodataset/cocoapi/pull/332/
                    tps = np.logical_and(dtm >= 0, np.logical_not(dtIg))
                    fps = np.logical_and(dtm < 0, np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            # https://github.com/cocodataset/cocoapi/pull/405
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {: 0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps_crowd():
            # Adapted from https://github.com/Jeff-sjtu/CrowdPose
            # @article{li2018crowdpose,
            #   title={CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark},
            #   author={Li, Jiefeng and Wang, Can and Zhu, Hao and Mao, Yihuan and Fang, Hao-Shu and Lu, Cewu},
            #   journal={arXiv preprint arXiv:1812.00324},
            #   year={2018}
            # }
            stats = np.zeros((9,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(0, maxDets=20)
            stats[4] = _summarize(0, maxDets=20, iouThr=.5)
            stats[5] = _summarize(0, maxDets=20, iouThr=.75)
            type_result = self.get_type_result(first=0.2, second=0.8)

            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | type={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision'
            typeStr = '(AP)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1])
            print(iStr.format(titleStr, typeStr, iouStr, 'easy', 20, type_result[0]))
            print(iStr.format(titleStr, typeStr, iouStr, 'medium', 20, type_result[1]))
            print(iStr.format(titleStr, typeStr, iouStr, 'hard', 20, type_result[2]))
            stats[6] = type_result[0]
            stats[7] = type_result[1]
            stats[8] = type_result[2]

            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')

            # ----------------------------------------------------------------
            # all_mean = 0
            # for i in range(10):
            #     temp_mean = _summarize(1, maxDets=20, iouThr=0.5 + i * 0.05)
            #     print(temp_mean)
            #     all_mean += temp_mean
            # print(all_mean / 10)
            #-----------------------------------------------------------------
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints_crowd':
            summarize = _summarizeKps_crowd
        elif 'keypoints' in iouType:
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()

    def get_type_result(self,  first=0.01, second=0.85):
        gt_file, resfile = self.anno_file
        easy, mid, hard = self.split(gt_file, first, second)
        res = []
        nullwrite = NullWriter()
        oldstdout = sys.stdout
        sys.stdout = nullwrite
        for curr_type in [easy, mid, hard]:
            curr_list = curr_type
            self.params.imgIds = curr_list
            self.evaluate()
            self.accumulate()
            score = self.eval['precision'][:, :, :, 0, :]
            res.append(round(np.mean(score), 4))
        sys.stdout = oldstdout
        return res


    def split(serlf, gt_file, first=0.01, second=0.85):
        import json
        data = json.load(
            open(gt_file, 'r'))
        easy = []
        mid = []
        hard = []
        for item in data['images']:
            if item['crowdIndex'] < first:
                easy.append(item['id'])
            elif item['crowdIndex'] < second:
                mid.append(item['id'])
            else:
                hard.append(item['id'])
        return easy, mid, hard

# =====================================================================================
    def my_evaluate(self, my_vis_thr):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running MY image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif 'keypoints' in p.iouType:
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.my_evaluateImg
        maxDet = p.maxDets[-1]
        # self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet, vis_thr)
        #          for catId in catIds
        #          for areaRng in p.areaRng
        #          for imgId in p.imgIds
        #      ]

        # per_img_recall_list = [evaluateImg(imgId, catId, maxDet, vis_thr)
        #          for catId in catIds
        #          for areaRng in p.areaRng
        #          for imgId in p.imgIds
        #      ]
        # ------------------------------------------------------------------------
        all_recall_list = np.array([[0 for __ in range(11)] for _ in range(len(p.iouThrs))])
        all_recall_sum = np.array([0 for _ in range(len(p.iouThrs))], dtype=np.float32)
        all_recall_rate_list = [[] for _ in range(len(p.iouThrs))]
        maxdist = 0
        mindist = 100000
        for catId in catIds:
            for imgId in p.imgIds:
                per_img_recall_list, temp_recall_sum, recall_rate_list, min_dist, max_dist = evaluateImg(
                    imgId, 
                    catId, 
                    maxDet, 
                    my_vis_thr
                )
                maxdist = max(maxdist, max_dist)
                mindist = min(mindist, min_dist)
                all_recall_list += per_img_recall_list
                all_recall_sum += temp_recall_sum
                for i in range(len(p.iouThrs)):
                    all_recall_rate_list[i].extend(recall_rate_list[i])
                
        print(f'max_dist : {maxdist}, min_dist : {mindist}')
        # all_recall_rate_list = np.array(all_recall_rate_list[0], dtype=np.float32)
        all_recall_rate_list = all_recall_rate_list[0]
        all_recall_rate_list.sort()
        # all_recall_rate_list = all_recall_rate_list[::-1]
        # print(all_recall_rate_list)

        postivate_length = 1605
        all_recall_rate_list = all_recall_rate_list[:postivate_length]
        temp_all_recall_sum = 0
        for item in all_recall_rate_list:
            temp_all_recall_sum += item[1]
        print(temp_all_recall_sum / postivate_length)
        # with open('/home/chenbeitao/data/code/COCO-Statistic/Data-Statistic/vis.txt', 'a') as fd:
        #     fd.write(f'{my_vis_thr} : {temp_all_recall_sum / postivate_length * 100:>.4f} |')
        
        all_num = all_recall_list.sum(axis=1).reshape(-1, 1)
        print('all_num : ', all_num)

        print(all_recall_list)
        print('*'*100)
        print('all_recall_rate : ', np.around(all_recall_sum / all_num.T, 4))
        print('*'*100)
        # all_recall_list = all_recall_list.astype(np.float32)
        # all_recall_list /= all_num
        np.set_printoptions(suppress=True)
        print(np.around(all_recall_list / all_num, 6))
        # ------------------------------------------------------------------------

        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))


    def my_evaluateImg(self, imgId, catId, maxDet, my_vis_thr):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        recall_ans_list = np.array([[0 for __ in range(11)] for _ in range(len(p.iouThrs))])
        temp_recall_sum = np.array([0 for _ in range(len(p.iouThrs))], dtype=np.float32)
        recall_rate_list = [[] for _ in range(len(p.iouThrs))]

        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            # return None
            return recall_ans_list, temp_recall_sum, recall_rate_list, 10000, 0

        for g in gt:
            if 'area' not in g or not self.use_area:
                tmp_area = g['bbox'][2] * g['bbox'][3] * 0.53
            else:
                tmp_area = g['area']
            # if g['ignore'] or (tmp_area < aRng[0] or tmp_area > aRng[1]):
            #     g['_ignore'] = 1
            # else:
            #     g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        # gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        # gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d[self.score_key] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        # ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]
        ious = self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        # https://github.com/cocodataset/cocoapi/pull/332/
        gtm  = np.ones((T,G)) * -1
        gtmatch_list = np.ones((T, G), dtype=np.int32) * -1
        dtm  = np.ones((T,D)) * -1
        # gtIg = np.array([g['_ignore'] for g in gt])
        # dtIg = np.zeros((T,D))

        # if len(ious) == 0:
        #     print('A',end='')
        #     self._calculate_per_person_keypoint_recal(
        #         p=p,
        #         gt=gt,
        #         gtmatch_list=gtmatch_list,
        #         recall_ans_list=recall_rate_list,
        #         recall_rate_list=recall_rate_list,
        #         temp_recall_sum=temp_recall_sum,
        #         vis_thr=vis_thr
        #     )
            
        if len(ious):
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>=0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        # since all the rest of g's are ignored as well because of the prior sorting
                        # if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                        #     break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    # dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
                    gtmatch_list[tind, m] = dind

        # if len(ious) == 0:
        #     # return [[0 for j in gt] for _ in p.iouThrs]
        #     if len(gt) == 0:
        #         pass
        #     elif len(dt) == 0:
        #         count = 0
        #         for g in gt:
        #             count += (np.sum(np.array(g['keypoints'][2::3]) > 0) == 0)
        #         if count == len(gt):
        #             print(count)
        #         else:
        #             print('@' * 100)

            
        min_dist, max_dist = self._calculate_per_person_keypoint_recal(
            p=p,
            gt=gt,
            dt=dt,
            gtmatch_list=gtmatch_list,
            recall_ans_list=recall_ans_list,
            recall_rate_list=recall_rate_list,
            temp_recall_sum=temp_recall_sum,
            my_vis_thr=my_vis_thr
        )
            # for tind, t in enumerate(p.iouThrs):
            #     for gind, g in enumerate(gt):
            #         # if data_type == 'coco'
            #         complete_index = self._calculate_coco_complete(np.array(g['keypoints']).reshape(17, 3))
            #         if complete_index >= 0.5:
            #         # if complete_index != 0:
            #             continue

            #         gt_kpt = np.array(g['keypoints'][2::3])
            #         gt_kpt = (gt_kpt > 0)
            #         all_num_kpt = gt_kpt.sum()
            #         if all_num_kpt == 0:
            #             continue

            #         if gtmatch_list[tind, gind] == -1:
            #             recall_ans_list[tind][0] += 1
            #             recall_rate_list[tind].append([1, 0])
            #             continue
            #         dt_kpt = np.array(dt[gtmatch_list[tind, gind]]['keypoints'][2::3])
            #         dt_kpt = (dt_kpt > vis_thr)
            #         # dt_kpt = (dt_kpt > 0)
            #         # correct = (gt_kpt == dt_kpt).sum()
            #         correct = (((gt_kpt == True).astype(int) +  (gt_kpt == dt_kpt).astype(int)) == 2).sum()                    
            #         person_kpt_recall = correct / all_num_kpt

            #         temp_recall_sum[tind] += person_kpt_recall
            #         recall_rate_list[tind].append([complete_index, person_kpt_recall])
            #         recall_ans_list[tind][int(person_kpt_recall * 10)] += 1
        
        return recall_ans_list, temp_recall_sum, recall_rate_list, min_dist, max_dist
        # set unmatched detections outside of area range to ignore
        # a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        # dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        # return {
        #         'image_id':     imgId,
        #         'category_id':  catId,
        #         'aRng':         aRng,
        #         'maxDet':       maxDet,
        #         'dtIds':        [d['id'] for d in dt],
        #         'gtIds':        [g['id'] for g in gt],
        #         'dtMatches':    dtm,
        #         'gtMatches':    gtm,
        #         'dtScores':     [d[self.score_key] for d in dt],
        #         'gtIgnore':     gtIg,
        #         'dtIgnore':     dtIg,
        #     }

    def _calculate_per_person_keypoint_recal(
        self, 
        p, gt, dt, 
        gtmatch_list, 
        recall_ans_list, recall_rate_list, temp_recall_sum,
        my_vis_thr
    ):

        mindist = 10000
        maxdist = 0
        for tind, t in enumerate(p.iouThrs):
            for gind, g in enumerate(gt):
                # if data_type == 'coco'
                complete_index = self._calculate_coco_complete(np.array(g['keypoints']).reshape(17, 3))

                # ---------------------------------------------------------------------
                ''' filter some incomplete images '''
                # if complete_index >= 0.5:
                # if complete_index < 0.5:
                # # if complete_index != 0:
                #     continue
                # ---------------------------------------------------------------------

                gt_kpt = np.array(g['keypoints'][2::3])
                gt_kpt = (gt_kpt > 0)
                all_num_kpt = gt_kpt.sum()
                if all_num_kpt == 0:
                    continue

                if gtmatch_list[tind, gind] == -1:
                    recall_ans_list[tind][0] += 1
                    recall_rate_list[tind].append([1, 0])

                    # ---------------------------------------------------------------------
                    ''' show bad case '''
                    # if t == 0.5:
                    #     self._show_bad_case(g, dt[gtmatch_list[tind, gind]], bad_flag=True)
                    
                    continue
                # if tind == 0:
                # self._show_bad_case(g, dt[gtmatch_list[tind, gind]], bad_flag=False)
                    # ---------------------------------------------------------------------

                dt_kpt = np.array(dt[gtmatch_list[tind, gind]]['keypoints'][2::3])
                
                IQ = 2
                if IQ == 1:
                    # =================================================================
                    '''
                        calculate detection rate by vis or distance between pose pair
                    by using TP / (TP + FN)
                    '''

                    # -----------------------------------------------------------------
                    ''' directly get vis'''
                    dt_kpt = (dt_kpt > my_vis_thr)                
                    # -----------------------------------------------------------------
                    ''' calculate keypoints distance between gt and dt to get vis'''
                    # min_dist, max_dist, dist_flag = self._dist_judge_vis(
                    #     g['keypoints'], 
                    #     dt[gtmatch_list[tind, gind]]['keypoints'],
                    #     g['bbox'],
                    #     my_vis_thr
                    # )
                    # mindist = min(mindist, min_dist)
                    # maxdist = max(maxdist, max_dist)
                    # dt_kpt = dist_flag
                    # -----------------------------------------------------------------

                    # dt_kpt = (dt_kpt > 0)
                    # correct = (gt_kpt == dt_kpt).sum()
                    # correct = (((gt_kpt == True).astype(int) +  (gt_kpt == dt_kpt).astype(int)) == 2).sum()                    
                    correct = ((gt_kpt == True) & (gt_kpt == dt_kpt)).sum()
                    person_kpt_recall = correct / all_num_kpt
                    # =================================================================
                elif IQ == 2:
                    # =================================================================
                    '''
                        calculate detection rate by vis or distance between pose pair
                    by using TP / (TP + FP/2 + FN/2)
                    '''
                    
                    # -----------------------------------------------------------------
                    ''' directly get vis'''
                    person_kpt_recall = self._get_detection_rate2(
                        g, 
                        dt[gtmatch_list[tind, gind]], 
                        my_vis_thr
                    )
                    # -----------------------------------------------------------------
                    # =================================================================

                temp_recall_sum[tind] += person_kpt_recall
                recall_rate_list[tind].append([complete_index, person_kpt_recall])
                recall_ans_list[tind][int(person_kpt_recall * 10)] += 1

        return mindist, maxdist

    def _get_detection_rate2(self, g, d, my_vis_thr):
        assert type(g) == dict
        assert type(d) == dict

        # d = dt[gtm[tind, gind]]
        dt_kpt = np.array(d['keypoints']).reshape(17, 3)
        gt_kpt = np.array(g['keypoints']).reshape(17, 3)
        TPindex = ((gt_kpt[:, 2] > 0) & (dt_kpt[:, 2] > my_vis_thr))
        FPindex = ((gt_kpt[:, 2] == 0) & (dt_kpt[:, 2] > my_vis_thr))
        FNindex = ((gt_kpt[:, 2] > 0) & (dt_kpt[:, 2] <= my_vis_thr))
        PNindex = ((gt_kpt[:, 2] == 0) & (dt_kpt[:, 2] <= my_vis_thr))
        TP = TPindex.sum()
        FP = FPindex.sum()
        FN = FNindex.sum()
        PN = PNindex.sum()
        # rate = gtm_ious[tind, gind] * TP / (TP + FP / 2 + FN / 2)
        # rate = TP / (TP + FP / 2 + FN / 2)
        # rate = TP / (TP + FP + FN / 2)
        rate = TP / (TP + FP + FN + PN)
        print((TP + FP + FN + PN))

        # rate =  TP / (TP + FN)
        # print((gt_kpt[:, 2] > 0).sum(), end=' ')

        # print((TP + FP / 2 + FN / 2))
        # regress_rate = gtm_ious[tind, gind] 
        # TPfactor = dt_kpt[TPindex, 2].mean() if (TPindex.sum() != 0) else 0
        # FPfactor = dt_kpt[FPindex, 2].mean() if (FPindex.sum() != 0) else 0
        # FNfactor = dt_kpt[FNindex, 2].mean() if (FNindex.sum() != 0) else 0
        # check_rate = TPfactor * TP / (TP*TPfactor + FP*(1 - FPfactor) + FN*FNfactor + np.spacing(1))
        # check_rate = TPfactor * TP / (TP*TPfactor + FN*FNfactor + np.spacing(1))
        # print(check_rate)
        # if tind == 0:jndex, 2].mean(),dt_kpt[FPindex, 2].mean(), dt_kpt[FNindex, 2].mean(), check_rate)
        # rate = regress_rate * check_rate
        # rate = check_rate

        # ans_pq[tind].append(rate)
        return rate


    def _show_bad_case(self, g, dt, bad_flag=True):
        import os
        import cv2

        image_path = '/home/chenbeitao/data/code/Test/filter-image/images/all-image'
        coco_origin_val_dir = '/mnt/hdd3/wangxuanhan/datasets/coco/val2017'
        file_name = str(g['image_id']).rjust(12, '0')+'.jpg'
        bbox = g['bbox']
        if os.path.exists(os.path.join(image_path, file_name)):
            img = cv2.imread(os.path.join(image_path, file_name))
        else:
            img = cv2.imread(os.path.join(coco_origin_val_dir, file_name))
        
        if bad_flag:
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0, 0, 255), 4)
            kpts = g['keypoints']
            x, y, vis = kpts[0::3], kpts[1::3], kpts[2::3]
            for i in range(17):
                if vis[i] > 0:
                    cv2.circle(img, ((int)(x[i]), (int)(y[i])), 1, (0, 255, 0), 8)
        else:
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (255, 0, 0), 1)
            kpts = dt['keypoints']
            gt_kpts = g['keypoints']
            # print(type(kpts))
            # print(kpts)
            x, y, vis = kpts[0::3], kpts[1::3], gt_kpts[2::3]
            for i in range(17):
                # if vis[i] == 1:
                #     cv2.circle(img, (x[i], y[i]), 1, (0, 0, 255), 4)
                # elif vis[i] == 2:
                if vis[i] > 0:
                    cv2.circle(img, ((int)(x[i]), (int)(y[i])), 1, (10,215,255), 4)
        cv2.imwrite(os.path.join(image_path, file_name), img)
    
    def _add_annotations_to_image(self, ann_list, img, file_name):
        import cv2
        for ann in ann_list:
            kpts = ann['keypoints']
            bbox = ann['bbox']
            x, y, vis = kpts[0::3], kpts[1::3], kpts[2::3]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (255, 0, 0), 2)
            for i in range(17):
                if vis[i] == 1:
                    cv2.circle(img, (x[i], y[i]), 1, (0, 0, 255), 4)
                elif vis[i] == 2:
                    cv2.circle(img, (x[i], y[i]), 1, (0, 255, 0), 4)

    def _dist_judge_vis(self, gt_kpt : list, dt_kpt : list, bbox : list, vis_thr : float):
        gt_kpt = np.array(gt_kpt).reshape(17, 3)
        dt_kpt = np.array(dt_kpt).reshape(17, 3)

        assert gt_kpt.shape == (17, 3)
        assert dt_kpt.shape == (17, 3)

        dpi = np.sqrt(bbox[2] * bbox[3])
        idx = (gt_kpt[:, 2] > 0)
        dist = np.sqrt(np.sum((gt_kpt[idx][:, :2] - dt_kpt[idx][:, :2]) ** 2, axis=1)) / dpi
        
        # print(gt_kpt[idx].shape)
        # print(gt_kpt[idx][:, :2].shape)
        # print(np.sum((gt_kpt[idx][:, :2] - dt_kpt[idx][:, :2]) ** 2, axis=1).shape)
        # print(dist.shape)
        assert type(dist) == np.ndarray
        assert dist.shape[0] == gt_kpt[idx].shape[0]
        assert len(dist.shape) == 1

        dist_flag = np.zeros(dt_kpt[:, 2].shape, dtype=np.int32)
        # print(dist_flag.shape, dist.shape)
        dist_flag[idx] = (dist <= vis_thr)
        # print(dist.mean())
        assert len(dist_flag.shape) == 1

        return dist.min(), dist.max(), dist_flag
        

    def _calculate_body(self, kpts):
        assert len(kpts.shape) == 2

        all_num = kpts.shape[0]
        unvis_kpts = np.sum(kpts[:, 2] == 0)

        return unvis_kpts / all_num

    def _calculate_torso(self, kpts):
        return self._calculate_body(kpts)

    def _calculate_head(self, kpts):
        return self._calculate_body(kpts)

    def _calculate_arm(self, kpts):
        return self._calculate_body(kpts)

    def _calculate_leg(self, kpts):
        return self._calculate_body(kpts)

    def _calculate_all(self, complete_torso, complete_body : list):
        ans = 0.5 * complete_torso
        for item in complete_body:
            ans += 0.1 * item
        
        return ans

    def _caluculate_crowdpose_complete(self, kpts, data_type='crowdpose'):
        # if pre_torso_judge(np.vstack((kpts[0], kpts[1], kpts[6], kpts[7]))):
        #     return 1

        complete_torso = self._calculate_torso(np.vstack((kpts[0], kpts[1], kpts[6], kpts[7])))
        complete_head = self._calculate_head(np.vstack((kpts[12], kpts[13])))
        complete_left_arm = self._calculate_arm(np.vstack((kpts[3], kpts[5])))
        complete_right_arm = self._calculate_arm(np.vstack((kpts[2], kpts[4])))
        complete_left_leg = self._calculate_leg(np.vstack((kpts[9], kpts[11])))
        complete_right_leg = self._calculate_leg(np.vstack((kpts[8], kpts[10])))

        complete_index = self._calculate_all(
            complete_torso,
            [complete_head, complete_left_arm, complete_right_arm, complete_left_leg, complete_right_leg]    
        )

        return complete_index

    def _calculate_coco_complete(self, kpts, data_type='coco'):
        # if pre_torso_judge(np.vstack((kpts[5], kpts[6], kpts[11], kpts[12]))):
        #     return 1

        complete_torso = self._calculate_torso(np.vstack((kpts[5], kpts[6], kpts[11], kpts[12])))
        complete_head = self._calculate_head(np.vstack((kpts[0], kpts[1], kpts[2], kpts[3], kpts[4])))
        complete_left_arm = self._calculate_arm(np.vstack((kpts[7], kpts[9])))   
        complete_right_arm = self._calculate_arm(np.vstack((kpts[8], kpts[10])))
        complete_left_leg = self._calculate_leg(np.vstack((kpts[13], kpts[15])))
        complete_right_leg = self._calculate_leg(np.vstack((kpts[14], kpts[16])))

        complete_index = self._calculate_all(
            complete_torso, 
            [complete_head, complete_left_arm, complete_right_arm, complete_left_leg, complete_right_leg],
        )

        return complete_index






    def computeOks_PQ(self, imgId, catId, my_vis_thr):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d[self.score_key] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            self.RQ_IoU[(imgId, catId)] = None
            return []
        ious = np.zeros((len(dts), len(gts)))
        rq_ious = np.zeros((len(dts), len(gts)))
        sigmas = self.sigmas
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            if p.iouType == 'keypoints_wholebody':
                body_gt = gt['keypoints']
                foot_gt = gt['foot_kpts']
                face_gt = gt['face_kpts']
                lefthand_gt = gt['lefthand_kpts']
                righthand_gt = gt['righthand_kpts']
                wholebody_gt = body_gt + foot_gt + face_gt + lefthand_gt + righthand_gt
                g = np.array(wholebody_gt)
            elif p.iouType == 'keypoints_foot':
                g = np.array(gt['foot_kpts'])
            elif p.iouType == 'keypoints_face':
                g = np.array(gt['face_kpts'])
            elif p.iouType == 'keypoints_lefthand':
                g = np.array(gt['lefthand_kpts'])
            elif p.iouType == 'keypoints_righthand':
                g = np.array(gt['righthand_kpts'])
            else:
                g = np.array(gt['keypoints'])

            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                if p.iouType == 'keypoints_wholebody':
                    body_dt = dt['keypoints']
                    foot_dt = dt['foot_kpts']
                    face_dt = dt['face_kpts']
                    lefthand_dt = dt['lefthand_kpts']
                    righthand_dt = dt['righthand_kpts']
                    wholebody_dt = body_dt + foot_dt + face_dt + lefthand_dt + righthand_dt
                    d = np.array(wholebody_dt)
                elif p.iouType == 'keypoints_foot':
                    d = np.array(dt['foot_kpts'])
                elif p.iouType == 'keypoints_face':
                    d = np.array(dt['face_kpts'])
                elif p.iouType == 'keypoints_lefthand':
                    d = np.array(dt['lefthand_kpts'])
                elif p.iouType == 'keypoints_righthand':
                    d = np.array(dt['righthand_kpts'])
                else:
                    d = np.array(dt['keypoints'])

                xd = d[0::3]; yd = d[1::3]
                if k1>0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)

                if self.use_area:
                    e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
                else:
                    tmparea = gt['bbox'][3] * gt['bbox'][2] * 0.53
                    e = (dx**2 + dy**2) / vars / (tmparea+np.spacing(1)) / 2
                
                rq_e = copy.copy(e)
                if k1 > 0:
                    e=e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]

                #--------------------------------------------
                vd = d[2::3]
                FPindex = ((vg > 0) & (vd > my_vis_thr))
                FP = 0
                
                # if k1>0:
                #     # measure the per-keypoint distance if keypoints visible
                #     dx = xd - xg
                #     dy = yd - yg
                # else:
                #     # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                #     z = np.zeros((k))
                #     dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                #     dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
                # if self.use_area:
                #     rq_e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
                # else:
                #     tmparea = gt['bbox'][3] * gt['bbox'][2] * 0.53
                #     rq_e = (dx**2 + dy**2) / vars / (tmparea+np.spacing(1)) / 2
                
                # print(FPindex)
                # print(np.exp(-rq_e))
                
                if FPindex.sum() > 0:
                    # FP = FPindex.sum()
                    rq_e = rq_e[FPindex]
                    # print(rq_e.shape[0])
                    rq_ious[i, j] = np.sum(np.exp(-rq_e)) / rq_e.shape[0]
                else:
                    rq_ious[i, j] = 0
                # print(rq_ious[i, j])
        self.RQ_IoU[(imgId, catId)] = rq_ious
                #--------------------------------------------
                

        
        return ious

    def evaluate_PQ(self, my_vis_thr):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU

        # ---------------------------------------------------------------------------------            
        elif 'keypoints' in p.iouType:
            self.RQ_IoU = {}
            computeIoU = self.computeOks_PQ
            # computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId, my_vis_thr) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg_PQ
        maxDet = p.maxDets[-1]
        # self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
        #          for catId in catIds
        #          for areaRng in p.areaRng
        #          for imgId in p.imgIds
        #      ]

        
        ans_pq = [[] for _ in p.iouThrs]
        for imgId in p.imgIds:
            temp_ans_pq = evaluateImg(imgId, catIds[0], maxDet, my_vis_thr)
            for i in range(len(p.iouThrs)):
                ans_pq[i].extend(temp_ans_pq[i])
        ans_pq = np.array(ans_pq)
        print(ans_pq.shape)
        print(ans_pq.mean(axis=1))
        print(ans_pq.max(axis=1))
        print(ans_pq.min(axis=1))
        # ---------------------------------------------------------------------------------


        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))


    def evaluateImg_PQ(self, imgId, catId, maxDet, my_vis_thr):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        ans_pq = [[] for _ in p.iouThrs]
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return ans_pq

        for g in gt:
            if 'area' not in g or not self.use_area:
                tmp_area = g['bbox'][2] * g['bbox'][3] * 0.53
            else:
                tmp_area =g['area']

        # sort dt highest score first, sort gt ignore last
        # gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        # gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d[self.score_key] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId]
        rq_ious = self.RQ_IoU[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        # https://github.com/cocodataset/cocoapi/pull/332/
        gtm  = np.ones((T,G), dtype=np.int32) * -1
        gtm_ious = np.zeros((T, G))
        dtm  = np.ones((T,D)) * -1
        # gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if len(ious):
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    rq_iou = 0
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>=0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        # since all the rest of g's are ignored as well because of the prior sorting
                        # if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                        #     break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        rq_iou = rq_ious[dind, gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    # dtIg[tind,dind] = gtIg[m]
                    # dtm[tind,dind]  = gt[m]['id']
                    # gtm[tind,m]     = d['id']
                    dtm[tind, dind] = gind
                    gtm[tind, m] = dind
                    gtm_ious[tind, m] = rq_iou
                    # gtm_ious[tind, m] = iou

        if len(ious) == 0:
            # return [[0 for j in gt] for _ in p.iouThrs]
            if len(gt) == 0:
                return ans_pq   
            elif len(dt) == 0:
                count = 0
                for g in gt:
                    count += (np.sum(np.array(g['keypoints'][2::3]) > 0) == 0)
                if count == len(gt):
                    return ans_pq

                return [[0 for j in gt] for _ in p.iouThrs]

        
        for tind, t in enumerate(p.iouThrs):
            for gind, g in enumerate(gt):
                if np.sum(np.array(g['keypoints'][2::3]) > 0) == 0:
                    continue
                d = dt[gtm[tind, gind]]
                dt_kpt = np.array(d['keypoints']).reshape(17, 3)
                gt_kpt = np.array(g['keypoints']).reshape(17, 3)
                TPindex = ((gt_kpt[:, 2] > 0) & (dt_kpt[:, 2] > my_vis_thr))
                FPindex = ((gt_kpt[:, 2] == 0) & (dt_kpt[:, 2] > my_vis_thr))
                FNindex = ((gt_kpt[:, 2] > 0) & (dt_kpt[:, 2] <= my_vis_thr))
                TP = TPindex.sum()
                FP = FPindex.sum()
                FN = FNindex.sum()
                # rate = gtm_ious[tind, gind] * TP / (TP + FP / 2 + FN / 2)
                rate = TP / (TP + FP / 2 + FN / 2)
                raise Exception

                # rate =   gtm_ious[tind, gind] * TP / (TP + FN)
                # rate = gtm_ious[tind, gind]
                # print((gt_kpt[:, 2] > 0).sum(), end=' ')

                # print((TP + FP / 2 + FN / 2))
                # regress_rate = gtm_ious[tind, gind] 
                # TPfactor = dt_kpt[TPindex, 2].mean() if (TPindex.sum() != 0) else 0
                # FPfactor = dt_kpt[FPindex, 2].mean() if (FPindex.sum() != 0) else 0
                # FNfactor = dt_kpt[FNindex, 2].mean() if (FNindex.sum() != 0) else 0
                # check_rate = TPfactor * TP / (TP*TPfactor + FP*(1 - FPfactor) + FN*FNfactor + np.spacing(1))
                # check_rate = TPfactor * TP / (TP*TPfactor + FN*FNfactor + np.spacing(1))
                # print(check_rate)
                # if tind == 0:jndex, 2].mean(),dt_kpt[FPindex, 2].mean(), dt_kpt[FNindex, 2].mean(), check_rate)
                # rate = regress_rate * check_rate
                # rate = check_rate

                ans_pq[tind].append(rate)
                # if tind == 0:
                #     print(check_rate)

        return ans_pq

# =====================================================================================




class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif 'keypoints' in iouType:
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
