
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import register_cmap

cdict = {'red': [[0.0, 0.19215686274509805, 0.19215686274509805], [0.008403361344537815, 0.19607843137254902, 0.19607843137254902], [0.01680672268907563, 0.2, 0.2], [0.025210084033613446, 0.2, 0.2], [0.03361344537815126, 0.20392156862745098, 0.20392156862745098], [0.04201680672268908, 0.20784313725490197, 0.20784313725490197], [0.05042016806722689, 0.20784313725490197, 0.20784313725490197], [0.058823529411764705, 0.21176470588235294, 0.21176470588235294], [0.06722689075630252, 0.21176470588235294, 0.21176470588235294], [0.07563025210084033, 0.2196078431372549, 0.2196078431372549], [0.08403361344537816, 0.22745098039215686, 0.22745098039215686], [0.09243697478991597, 0.23529411764705882, 0.23529411764705882], [0.10084033613445378, 0.24313725490196078, 0.24313725490196078], [0.1092436974789916, 0.25098039215686274, 0.25098039215686274], [0.11764705882352941, 0.25882352941176473, 0.25882352941176473], [0.12605042016806722, 0.26666666666666666, 0.26666666666666666], [0.13445378151260504, 0.27450980392156865, 0.27450980392156865], [0.14285714285714285, 0.2784313725490196, 0.2784313725490196], [0.15126050420168066, 0.2901960784313726, 0.2901960784313726], [0.15966386554621848, 0.2980392156862745, 0.2980392156862745], [0.16806722689075632, 0.3058823529411765, 0.3058823529411765], [0.17647058823529413, 0.3137254901960784, 0.3137254901960784], [0.18487394957983194, 0.3215686274509804, 0.3215686274509804], [0.19327731092436976, 0.32941176470588235, 0.32941176470588235], [0.20168067226890757, 0.33725490196078434, 0.33725490196078434], [0.21008403361344538, 0.34509803921568627, 0.34509803921568627], [0.2184873949579832, 0.35294117647058826, 0.35294117647058826], [0.226890756302521, 0.36470588235294116, 0.36470588235294116], [0.23529411764705882, 0.37254901960784315, 0.37254901960784315], [0.24369747899159663, 0.3803921568627451, 0.3803921568627451], [0.25210084033613445, 0.38823529411764707, 0.38823529411764707], [0.2605042016806723, 0.396078431372549, 0.396078431372549], [0.2689075630252101, 0.403921568627451, 0.403921568627451], [0.2773109243697479, 0.4117647058823529, 0.4117647058823529], [0.2857142857142857, 0.4196078431372549, 0.4196078431372549], [0.29411764705882354, 0.4235294117647059, 0.4235294117647059], [0.3025210084033613, 0.43137254901960786, 0.43137254901960786], [0.31092436974789917, 0.4392156862745098, 0.4392156862745098], [0.31932773109243695, 0.4470588235294118, 0.4470588235294118], [0.3277310924369748, 0.45098039215686275, 0.45098039215686275], [0.33613445378151263, 0.4588235294117647, 0.4588235294117647], [0.3445378151260504, 0.4666666666666667, 0.4666666666666667], [0.35294117647058826, 0.4745098039215686, 0.4745098039215686], [0.36134453781512604, 0.4823529411764706, 0.4823529411764706], [0.3697478991596639, 0.48627450980392156, 0.48627450980392156], [0.37815126050420167, 0.49411764705882355, 0.49411764705882355], [0.3865546218487395, 0.5019607843137255, 0.5019607843137255], [0.3949579831932773, 0.5098039215686274, 0.5098039215686274], [0.40336134453781514, 0.5176470588235295, 0.5176470588235295], [0.4117647058823529, 0.5254901960784314, 0.5254901960784314], [0.42016806722689076, 0.5333333333333333, 0.5333333333333333], [0.42857142857142855, 0.5411764705882353, 0.5411764705882353], [0.4369747899159664, 0.5490196078431373, 0.5490196078431373], [0.44537815126050423, 0.5568627450980392, 0.5568627450980392], [0.453781512605042, 0.5686274509803921, 0.5686274509803921], [0.46218487394957986, 0.5764705882352941, 0.5764705882352941], [0.47058823529411764, 0.5882352941176471, 0.5882352941176471], [0.4789915966386555, 0.6, 0.6], [0.48739495798319327, 0.6078431372549019, 0.6078431372549019], [0.4957983193277311, 0.6196078431372549, 0.6196078431372549], [0.5042016806722689, 0.6274509803921569, 0.6274509803921569], [0.5126050420168067, 0.6392156862745098, 0.6392156862745098], [0.5210084033613446, 0.6509803921568628, 0.6509803921568628], [0.5294117647058824, 0.6627450980392157, 0.6627450980392157], [0.5378151260504201, 0.6745098039215687, 0.6745098039215687], [0.5462184873949579, 0.6862745098039216, 0.6862745098039216], [0.5546218487394958, 0.6980392156862745, 0.6980392156862745], [0.5630252100840336, 0.7058823529411765, 0.7058823529411765], [0.5714285714285714, 0.7176470588235294, 0.7176470588235294], [0.5798319327731093, 0.7294117647058823, 0.7294117647058823], [0.5882352941176471, 0.7411764705882353, 0.7411764705882353], [0.5966386554621849, 0.7529411764705882, 0.7529411764705882], [0.6050420168067226, 0.7568627450980392, 0.7568627450980392], [0.6134453781512605, 0.7686274509803922, 0.7686274509803922], [0.6218487394957983, 0.7803921568627451, 0.7803921568627451], [0.6302521008403361, 0.792156862745098, 0.792156862745098], [0.6386554621848739, 0.8, 0.8], [0.6470588235294118, 0.8117647058823529, 0.8117647058823529], [0.6554621848739496, 0.8196078431372549, 0.8196078431372549], [0.6638655462184874, 0.8313725490196079, 0.8313725490196079], [0.6722689075630253, 0.8392156862745098, 0.8392156862745098], [0.680672268907563, 0.8470588235294118, 0.8470588235294118], [0.6890756302521008, 0.8549019607843137, 0.8549019607843137], [0.6974789915966386, 0.8666666666666667, 0.8666666666666667], [0.7058823529411765, 0.8745098039215686, 0.8745098039215686], [0.7142857142857143, 0.8862745098039215, 0.8862745098039215], [0.7226890756302521, 0.8941176470588236, 0.8941176470588236], [0.7310924369747899, 0.9058823529411765, 0.9058823529411765], [0.7394957983193278, 0.9137254901960784, 0.9137254901960784], [0.7478991596638656, 0.9254901960784314, 0.9254901960784314], [0.7563025210084033, 0.9333333333333333, 0.9333333333333333], [0.7647058823529411, 0.9450980392156862, 0.9450980392156862], [0.773109243697479, 0.9529411764705882, 0.9529411764705882], [0.7815126050420168, 0.9647058823529412, 0.9647058823529412], [0.7899159663865546, 0.9725490196078431, 0.9725490196078431], [0.7983193277310925, 0.984313725490196, 0.984313725490196], [0.8067226890756303, 0.9921568627450981, 0.9921568627450981], [0.8151260504201681, 1.0, 1.0], [0.8235294117647058, 1.0, 1.0], [0.8319327731092437, 1.0, 1.0], [0.8403361344537815, 1.0, 1.0], [0.8487394957983193, 1.0, 1.0], [0.8571428571428571, 1.0, 1.0], [0.865546218487395, 1.0, 1.0], [0.8739495798319328, 1.0, 1.0], [0.8823529411764706, 1.0, 1.0], [0.8907563025210085, 1.0, 1.0], [0.8991596638655462, 1.0, 1.0], [0.907563025210084, 1.0, 1.0], [0.9159663865546218, 1.0, 1.0], [0.9243697478991597, 1.0, 1.0], [0.9327731092436975, 1.0, 1.0], [0.9411764705882353, 1.0, 1.0], [0.9495798319327731, 1.0, 1.0], [0.957983193277311, 1.0, 1.0], [0.9663865546218487, 1.0, 1.0], [0.9747899159663865, 1.0, 1.0], [0.9831932773109243, 1.0, 1.0], [0.9915966386554622, 1.0, 1.0], [1.0, 1.0, 1.0]], 'green': [[0.0, 0.14901960784313725, 0.14901960784313725], [0.008403361344537815, 0.15294117647058825, 0.15294117647058825], [0.01680672268907563, 0.15294117647058825, 0.15294117647058825], [0.025210084033613446, 0.1568627450980392, 0.1568627450980392], [0.03361344537815126, 0.1568627450980392, 0.1568627450980392], [0.04201680672268908, 0.1568627450980392, 0.1568627450980392], [0.05042016806722689, 0.1607843137254902, 0.1607843137254902], [0.058823529411764705, 0.1607843137254902, 0.1607843137254902], [0.06722689075630252, 0.16470588235294117, 0.16470588235294117], [0.07563025210084033, 0.16470588235294117, 0.16470588235294117], [0.08403361344537816, 0.17254901960784313, 0.17254901960784313], [0.09243697478991597, 0.17647058823529413, 0.17647058823529413], [0.10084033613445378, 0.1843137254901961, 0.1843137254901961], [0.1092436974789916, 0.19215686274509805, 0.19215686274509805], [0.11764705882352941, 0.19607843137254902, 0.19607843137254902], [0.12605042016806722, 0.20392156862745098, 0.20392156862745098], [0.13445378151260504, 0.20784313725490197, 0.20784313725490197], [0.14285714285714285, 0.21176470588235294, 0.21176470588235294], [0.15126050420168066, 0.2196078431372549, 0.2196078431372549], [0.15966386554621848, 0.2235294117647059, 0.2235294117647059], [0.16806722689075632, 0.23137254901960785, 0.23137254901960785], [0.17647058823529413, 0.23529411764705882, 0.23529411764705882], [0.18487394957983194, 0.24313725490196078, 0.24313725490196078], [0.19327731092436976, 0.24705882352941178, 0.24705882352941178], [0.20168067226890757, 0.2549019607843137, 0.2549019607843137], [0.21008403361344538, 0.25882352941176473, 0.25882352941176473], [0.2184873949579832, 0.26666666666666666, 0.26666666666666666], [0.226890756302521, 0.27450980392156865, 0.27450980392156865], [0.23529411764705882, 0.2784313725490196, 0.2784313725490196], [0.24369747899159663, 0.28627450980392155, 0.28627450980392155], [0.25210084033613445, 0.2901960784313726, 0.2901960784313726], [0.2605042016806723, 0.29411764705882354, 0.29411764705882354], [0.2689075630252101, 0.2980392156862745, 0.2980392156862745], [0.2773109243697479, 0.30196078431372547, 0.30196078431372547], [0.2857142857142857, 0.3058823529411765, 0.3058823529411765], [0.29411764705882354, 0.3058823529411765, 0.3058823529411765], [0.3025210084033613, 0.30980392156862746, 0.30980392156862746], [0.31092436974789917, 0.3137254901960784, 0.3137254901960784], [0.31932773109243695, 0.3176470588235294, 0.3176470588235294], [0.3277310924369748, 0.3176470588235294, 0.3176470588235294], [0.33613445378151263, 0.3215686274509804, 0.3215686274509804], [0.3445378151260504, 0.3254901960784314, 0.3254901960784314], [0.35294117647058826, 0.32941176470588235, 0.32941176470588235], [0.36134453781512604, 0.3333333333333333, 0.3333333333333333], [0.3697478991596639, 0.3333333333333333, 0.3333333333333333], [0.37815126050420167, 0.33725490196078434, 0.33725490196078434], [0.3865546218487395, 0.3411764705882353, 0.3411764705882353], [0.3949579831932773, 0.3411764705882353, 0.3411764705882353], [0.40336134453781514, 0.34509803921568627, 0.34509803921568627], [0.4117647058823529, 0.34901960784313724, 0.34901960784313724], [0.42016806722689076, 0.35294117647058826, 0.35294117647058826], [0.42857142857142855, 0.3568627450980392, 0.3568627450980392], [0.4369747899159664, 0.3568627450980392, 0.3568627450980392], [0.44537815126050423, 0.3607843137254902, 0.3607843137254902], [0.453781512605042, 0.3686274509803922, 0.3686274509803922], [0.46218487394957986, 0.37254901960784315, 0.37254901960784315], [0.47058823529411764, 0.3803921568627451, 0.3803921568627451], [0.4789915966386555, 0.3843137254901961, 0.3843137254901961], [0.48739495798319327, 0.38823529411764707, 0.38823529411764707], [0.4957983193277311, 0.396078431372549, 0.396078431372549], [0.5042016806722689, 0.4, 0.4], [0.5126050420168067, 0.40784313725490196, 0.40784313725490196], [0.5210084033613446, 0.4117647058823529, 0.4117647058823529], [0.5294117647058824, 0.4196078431372549, 0.4196078431372549], [0.5378151260504201, 0.4235294117647059, 0.4235294117647059], [0.5462184873949579, 0.43137254901960786, 0.43137254901960786], [0.5546218487394958, 0.43529411764705883, 0.43529411764705883], [0.5630252100840336, 0.44313725490196076, 0.44313725490196076], [0.5714285714285714, 0.4470588235294118, 0.4470588235294118], [0.5798319327731093, 0.4549019607843137, 0.4549019607843137], [0.5882352941176471, 0.4588235294117647, 0.4588235294117647], [0.5966386554621849, 0.4666666666666667, 0.4666666666666667], [0.6050420168067226, 0.4666666666666667, 0.4666666666666667], [0.6134453781512605, 0.4745098039215686, 0.4745098039215686], [0.6218487394957983, 0.47843137254901963, 0.47843137254901963], [0.6302521008403361, 0.48627450980392156, 0.48627450980392156], [0.6386554621848739, 0.49411764705882355, 0.49411764705882355], [0.6470588235294118, 0.4980392156862745, 0.4980392156862745], [0.6554621848739496, 0.5058823529411764, 0.5058823529411764], [0.6638655462184874, 0.5137254901960784, 0.5137254901960784], [0.6722689075630253, 0.5215686274509804, 0.5215686274509804], [0.680672268907563, 0.5254901960784314, 0.5254901960784314], [0.6890756302521008, 0.5294117647058824, 0.5294117647058824], [0.6974789915966386, 0.5372549019607843, 0.5372549019607843], [0.7058823529411765, 0.5450980392156862, 0.5450980392156862], [0.7142857142857143, 0.5529411764705883, 0.5529411764705883], [0.7226890756302521, 0.5568627450980392, 0.5568627450980392], [0.7310924369747899, 0.5647058823529412, 0.5647058823529412], [0.7394957983193278, 0.5725490196078431, 0.5725490196078431], [0.7478991596638656, 0.5803921568627451, 0.5803921568627451], [0.7563025210084033, 0.5843137254901961, 0.5843137254901961], [0.7647058823529411, 0.592156862745098, 0.592156862745098], [0.773109243697479, 0.6, 0.6], [0.7815126050420168, 0.6039215686274509, 0.6039215686274509], [0.7899159663865546, 0.611764705882353, 0.611764705882353], [0.7983193277310925, 0.6196078431372549, 0.6196078431372549], [0.8067226890756303, 0.6274509803921569, 0.6274509803921569], [0.8151260504201681, 0.6352941176470588, 0.6352941176470588], [0.8235294117647058, 0.6470588235294118, 0.6470588235294118], [0.8319327731092437, 0.6588235294117647, 0.6588235294117647], [0.8403361344537815, 0.6627450980392157, 0.6627450980392157], [0.8487394957983193, 0.6745098039215687, 0.6745098039215687], [0.8571428571428571, 0.6862745098039216, 0.6862745098039216], [0.865546218487395, 0.6980392156862745, 0.6980392156862745], [0.8739495798319328, 0.7098039215686275, 0.7098039215686275], [0.8823529411764706, 0.7215686274509804, 0.7215686274509804], [0.8907563025210085, 0.7333333333333333, 0.7333333333333333], [0.8991596638655462, 0.7450980392156863, 0.7450980392156863], [0.907563025210084, 0.7568627450980392, 0.7568627450980392], [0.9159663865546218, 0.7647058823529411, 0.7647058823529411], [0.9243697478991597, 0.7764705882352941, 0.7764705882352941], [0.9327731092436975, 0.788235294117647, 0.788235294117647], [0.9411764705882353, 0.8, 0.8], [0.9495798319327731, 0.8117647058823529, 0.8117647058823529], [0.957983193277311, 0.8235294117647058, 0.8235294117647058], [0.9663865546218487, 0.8352941176470589, 0.8352941176470589], [0.9747899159663865, 0.8470588235294118, 0.8470588235294118], [0.9831932773109243, 0.8588235294117647, 0.8588235294117647], [0.9915966386554622, 0.8705882352941177, 0.8705882352941177], [1.0, 0.8823529411764706, 0.8823529411764706]], 'blue': [[0.0, 0.29411764705882354, 0.29411764705882354], [0.008403361344537815, 0.2980392156862745, 0.2980392156862745], [0.01680672268907563, 0.2980392156862745, 0.2980392156862745], [0.025210084033613446, 0.30196078431372547, 0.30196078431372547], [0.03361344537815126, 0.3058823529411765, 0.3058823529411765], [0.04201680672268908, 0.3058823529411765, 0.3058823529411765], [0.05042016806722689, 0.3058823529411765, 0.3058823529411765], [0.058823529411764705, 0.30980392156862746, 0.30980392156862746], [0.06722689075630252, 0.3137254901960784, 0.3137254901960784], [0.07563025210084033, 0.3137254901960784, 0.3137254901960784], [0.08403361344537816, 0.3215686274509804, 0.3215686274509804], [0.09243697478991597, 0.3254901960784314, 0.3254901960784314], [0.10084033613445378, 0.3333333333333333, 0.3333333333333333], [0.1092436974789916, 0.33725490196078434, 0.33725490196078434], [0.11764705882352941, 0.3411764705882353, 0.3411764705882353], [0.12605042016806722, 0.34901960784313724, 0.34901960784313724], [0.13445378151260504, 0.35294117647058826, 0.35294117647058826], [0.14285714285714285, 0.3568627450980392, 0.3568627450980392], [0.15126050420168066, 0.3607843137254902, 0.3607843137254902], [0.15966386554621848, 0.3686274509803922, 0.3686274509803922], [0.16806722689075632, 0.37254901960784315, 0.37254901960784315], [0.17647058823529413, 0.3803921568627451, 0.3803921568627451], [0.18487394957983194, 0.3843137254901961, 0.3843137254901961], [0.19327731092436976, 0.38823529411764707, 0.38823529411764707], [0.20168067226890757, 0.396078431372549, 0.396078431372549], [0.21008403361344538, 0.4, 0.4], [0.2184873949579832, 0.40784313725490196, 0.40784313725490196], [0.226890756302521, 0.4117647058823529, 0.4117647058823529], [0.23529411764705882, 0.41568627450980394, 0.41568627450980394], [0.24369747899159663, 0.4235294117647059, 0.4235294117647059], [0.25210084033613445, 0.42745098039215684, 0.42745098039215684], [0.2605042016806723, 0.43137254901960786, 0.43137254901960786], [0.2689075630252101, 0.43529411764705883, 0.43529411764705883], [0.2773109243697479, 0.4392156862745098, 0.4392156862745098], [0.2857142857142857, 0.44313725490196076, 0.44313725490196076], [0.29411764705882354, 0.4470588235294118, 0.4470588235294118], [0.3025210084033613, 0.45098039215686275, 0.45098039215686275], [0.31092436974789917, 0.4549019607843137, 0.4549019607843137], [0.31932773109243695, 0.4588235294117647, 0.4588235294117647], [0.3277310924369748, 0.4627450980392157, 0.4627450980392157], [0.33613445378151263, 0.4666666666666667, 0.4666666666666667], [0.3445378151260504, 0.47058823529411764, 0.47058823529411764], [0.35294117647058826, 0.4745098039215686, 0.4745098039215686], [0.36134453781512604, 0.47843137254901963, 0.47843137254901963], [0.3697478991596639, 0.47843137254901963, 0.47843137254901963], [0.37815126050420167, 0.4823529411764706, 0.4823529411764706], [0.3865546218487395, 0.48627450980392156, 0.48627450980392156], [0.3949579831932773, 0.49019607843137253, 0.49019607843137253], [0.40336134453781514, 0.49411764705882355, 0.49411764705882355], [0.4117647058823529, 0.4980392156862745, 0.4980392156862745], [0.42016806722689076, 0.5019607843137255, 0.5019607843137255], [0.42857142857142855, 0.5058823529411764, 0.5058823529411764], [0.4369747899159664, 0.5098039215686274, 0.5098039215686274], [0.44537815126050423, 0.5137254901960784, 0.5137254901960784], [0.453781512605042, 0.5176470588235295, 0.5176470588235295], [0.46218487394957986, 0.5215686274509804, 0.5215686274509804], [0.47058823529411764, 0.5254901960784314, 0.5254901960784314], [0.4789915966386555, 0.5294117647058824, 0.5294117647058824], [0.48739495798319327, 0.5333333333333333, 0.5333333333333333], [0.4957983193277311, 0.5372549019607843, 0.5372549019607843], [0.5042016806722689, 0.5411764705882353, 0.5411764705882353], [0.5126050420168067, 0.5450980392156862, 0.5450980392156862], [0.5210084033613446, 0.5490196078431373, 0.5490196078431373], [0.5294117647058824, 0.5529411764705883, 0.5529411764705883], [0.5378151260504201, 0.5568627450980392, 0.5568627450980392], [0.5462184873949579, 0.5607843137254902, 0.5607843137254902], [0.5546218487394958, 0.5647058823529412, 0.5647058823529412], [0.5630252100840336, 0.5686274509803921, 0.5686274509803921], [0.5714285714285714, 0.5725490196078431, 0.5725490196078431], [0.5798319327731093, 0.5764705882352941, 0.5764705882352941], [0.5882352941176471, 0.5803921568627451, 0.5803921568627451], [0.5966386554621849, 0.5843137254901961, 0.5843137254901961], [0.6050420168067226, 0.5843137254901961, 0.5843137254901961], [0.6134453781512605, 0.5882352941176471, 0.5882352941176471], [0.6218487394957983, 0.592156862745098, 0.592156862745098], [0.6302521008403361, 0.596078431372549, 0.596078431372549], [0.6386554621848739, 0.596078431372549, 0.596078431372549], [0.6470588235294118, 0.596078431372549, 0.596078431372549], [0.6554621848739496, 0.596078431372549, 0.596078431372549], [0.6638655462184874, 0.6, 0.6], [0.6722689075630253, 0.6, 0.6], [0.680672268907563, 0.6, 0.6], [0.6890756302521008, 0.6, 0.6], [0.6974789915966386, 0.6, 0.6], [0.7058823529411765, 0.6, 0.6], [0.7142857142857143, 0.6, 0.6], [0.7226890756302521, 0.6, 0.6], [0.7310924369747899, 0.6039215686274509, 0.6039215686274509], [0.7394957983193278, 0.6039215686274509, 0.6039215686274509], [0.7478991596638656, 0.6039215686274509, 0.6039215686274509], [0.7563025210084033, 0.6039215686274509, 0.6039215686274509], [0.7647058823529411, 0.6039215686274509, 0.6039215686274509], [0.773109243697479, 0.6039215686274509, 0.6039215686274509], [0.7815126050420168, 0.6039215686274509, 0.6039215686274509], [0.7899159663865546, 0.6078431372549019, 0.6078431372549019], [0.7983193277310925, 0.6078431372549019, 0.6078431372549019], [0.8067226890756303, 0.6078431372549019, 0.6078431372549019], [0.8151260504201681, 0.6078431372549019, 0.6078431372549019], [0.8235294117647058, 0.6, 0.6], [0.8319327731092437, 0.596078431372549, 0.596078431372549], [0.8403361344537815, 0.592156862745098, 0.592156862745098], [0.8487394957983193, 0.5882352941176471, 0.5882352941176471], [0.8571428571428571, 0.5803921568627451, 0.5803921568627451], [0.865546218487395, 0.5764705882352941, 0.5764705882352941], [0.8739495798319328, 0.5686274509803921, 0.5686274509803921], [0.8823529411764706, 0.5647058823529412, 0.5647058823529412], [0.8907563025210085, 0.5607843137254902, 0.5607843137254902], [0.8991596638655462, 0.5529411764705883, 0.5529411764705883], [0.907563025210084, 0.5490196078431373, 0.5490196078431373], [0.9159663865546218, 0.5450980392156862, 0.5450980392156862], [0.9243697478991597, 0.5411764705882353, 0.5411764705882353], [0.9327731092436975, 0.5333333333333333, 0.5333333333333333], [0.9411764705882353, 0.5294117647058824, 0.5294117647058824], [0.9495798319327731, 0.5215686274509804, 0.5215686274509804], [0.957983193277311, 0.5176470588235295, 0.5176470588235295], [0.9663865546218487, 0.5137254901960784, 0.5137254901960784], [0.9747899159663865, 0.5058823529411764, 0.5058823529411764], [0.9831932773109243, 0.5019607843137255, 0.5019607843137255], [0.9915966386554622, 0.49411764705882355, 0.49411764705882355], [1.0, 0.49019607843137253, 0.49019607843137253]]}

cmap = LinearSegmentedColormap('sunset', segmentdata=cdict, N=256)
register_cmap(name='sunset', cmap=cmap)