"""
dex.py
====================================
The module containing all the routines for evaluating DEX models.

Includes numerically improved QQ


"""
import xml.etree.ElementTree as ET
import numpy as np
import scipy.linalg as linalg


class ScaleValue:
    """
    Class for sotring the values for each scale of attributes.

    :param name: the name of the scale
    :param greoup: the group of the scale
    :param optim: a wrapper a for a PyTorch optimizer
    :param loss: an instance of a subclass of :class:`~pyro.infer.elbo.ELBO`.

    """

    def __init__(self, name, group, description=None, order=0):
        self.name = name
        self.group = group
        self.description = description
        self.order = order


class Scale:
    """
    Class for defining the scale of each attribute
    """

    ASC = 1
    DSC = 0

    def __init__(self, name, order, ordered):
        self.name = name
        self.order = order
        self.ordered = ordered
        self.scalevalue = {}

    def add_group(self, name, group):
        self.scalevalue[name] = group

    @staticmethod
    def parse(node):
        name = node.findall("NAME")[0].text.strip()
        ordered = node.findall("ORDERED")[0].text == "YES"
        order = Scale.ASC if node.findall("ORDER")[0].text == "ASC" else Scale.DSC

        obj = Scale(name, order, ordered)

        for order, sval in enumerate(node.findall("SCALEVALUE")):
            name = sval.findall("NAME")[0].text
            group = sval.findall("GROUP")
            if group:
                group = group[0].text
            else:
                group = None
            obj.add_group(name, ScaleValue(name, group, order=order))
        return obj


class Attribute:
    def __init__(self, name, description, scale, parent):
        self.name = name
        self.description = description
        self.scale = scale
        self.parentstr = parent
        self.parent = None
        self.child_attrs = []
        self.dex_function = None
        self.level = None
        self.helper_qq_interval()

    @staticmethod
    def parse(node, scales):
        name = node.findall("NAME")[0].text.strip()
        scale = scales[node.findall("SCALE")[0].text.strip()]
        desc = (
            node.findall("DESCRIPTION")[0].text.strip()
            if node.findall("DESCRIPTION")
            else None
        )
        parent = (
            node.findall("PARENT")[0].text.strip() if node.findall("PARENT") else None
        )
        function = (
            node.findall("FUNCTION")[0].text.strip()
            if node.findall("FUNCTION")
            else None
        )

        return Attribute(name, desc, scale, parent)

    def set_parent(self, attributes):
        if self.parentstr:
            self.parent = attributes[self.parentstr]
            attributes[self.parentstr].child_attrs.append(self)


    def helper_qq_interval(self, minv=1, maxv=None):
        if maxv is None:
            maxv = len(self.scale.scalevalue)

        self.qq_list = np.linspace(minv, maxv, len(self.scale.scalevalue))

    def map_qq(self, val):

        ind = np.argwhere(np.array(list(self.scale.scalevalue.keys())) == val).flatten()

        #         print(self.name, ind, val)

        if len(ind) != 1:
            raise Exception("Multiple mappings")
        return self.qq_list[ind[0]]

    def get_QQ_map(self, minv=1, maxv=None):
        if maxv is None:
            maxv = len(self.scale.scalevalue)

        qq_list = np.linspace(minv, maxv, len(self.scale.scalevalue))

        rv = {}
        for i, k in enumerate(self.scale.scalevalue.keys()):
            rv[k] = qq_list[i]

        return rv

    def set_function(self, dex_function):
        self.dex_function = dex_function


class DEXFunction:
    def __init__(self, name, attribute):
        self.name = name
        self.attr_list = []
        self.rules = {}
        self.rules_QQ = {}
        self.level = 0
        self.my_attribute = attribute
        self.output_values = []
        self.output_values_QQ = []
        self.kc = {}
        self.nc = {}

        self.w = 0

    def rules_to_QQ(self):
        for a in self.attr_list:
            k = a.name
            a.helper_qq_interval()
            self.rules_QQ[k] = np.array(list(map(a.map_qq, self.rules[k])))

        self.output_values_QQ = np.array(
            list(map(self.my_attribute.map_qq, self.output_values))
        )

        self.__vals = np.array([*self.rules_QQ.values()]).T

    def set_level(self):
        self.level = max([a.level for a in self.attr_list])

    def __calc_g_interval(self, A, w):
        num_samp = 10
        Ac = []
        for s in range(A.shape[1] - 1):
            low = np.min(A[:, s]) - 0.5
            high = np.max(A[:, s]) + 0.5
            Ac.append(np.linspace(low, high, num_samp))

        Ac = np.meshgrid(*Ac)
        Ac.append(np.ones(Ac[0].shape))
        Ac = np.array(Ac).T
        xx = Ac @ w
        return np.max(xx), np.min(xx)

    def kcnc(self):
        un = np.unique(self.output_values_QQ)

        A = []
        for k in self.rules_QQ:
            A.append(self.rules_QQ[k])
        A.append(np.ones(self.output_values_QQ.shape))

        A = np.vstack(A).T
        self.w, _, _, _ = linalg.lstsq(A, self.output_values_QQ)

        for c in un:
            ind = np.where(self.output_values_QQ == c)[0]
            g = A[ind, :] @ self.w
            minc, maxc = self.__calc_g_interval(A[ind, :], self.w)
            kc = 1 / (maxc - minc)
            nc = c + 0.5 - kc * maxc

            self.kc[c] = kc
            self.nc[c] = nc

    def evaluate_QQ(self, **input):
        A = []
        for attr in self.attr_list:
            if attr.name not in input:
                raise Exception("Missing value for %s" % attr.name)
            if ((not isinstance(input[attr.name], str)) and '*' in input[attr.name].astype(str)) or (isinstance(input[attr.name], str) and input[attr.name] == "*"):
                A.append(np.unique(self.rules_QQ[attr.name]))
            else:
                #                 A.append(np.array([attr.map_qq(input[attr.name])]).flatten())
                A.append(np.array([input[attr.name]]).flatten())

        A = np.array(np.meshgrid(*A)).T.reshape(-1, len(A))
        rval = []
        for row in A:
            data = dict(zip([o.name for o in self.attr_list], row))
            r = self.__evaluate_QQ(**data)
            rval.append(r[self.my_attribute.name])

        return {self.my_attribute.name: np.unique(np.array(rval))}

    def __evaluate_QQ(self, **input):
        exec_ind = None
        AttrVals = []
        """
        for attr in self.attr_list:
            if attr.name not in input:
                raise Exception('Missing value for %s' % attr.name)

            ind = []
            AttrVals.append(input[attr.name])
            for a_val in np.array([input[attr.name]]).flatten():
                inter_ind = np.argwhere(self.rules_QQ[attr.name] == np.round(a_val)).flatten()
                ind = np.union1d(ind, inter_ind).astype(int)

            if exec_ind is None:
                exec_ind = ind
            else:
                exec_ind = np.intersect1d(exec_ind,ind).flatten()

        A = AttrVals
        """

        A = [input[key] for key in self.rules_QQ.keys()]
        # for k in f.rules_QQ.keys():
        #     in_data.append(demo[k])

        #         print(self.name,A)
#         exec_ind = np.where((vals == np.array(A).round()).all(axis=1))[0]
        exec_ind = (self.__vals == np.array(A).round()).all(axis=1).nonzero()[0]

        if len(exec_ind) != 1:
            raise Exception(
                "Wrong number of rules executed %s for rule %s" % (exec_ind, self.name)
            )

        A.append(1)
        #         A = np.vstack(A).T

        c = self.output_values_QQ[exec_ind[0]]
        g = A @ self.w


        retVal = self.kc[c] * g + self.nc[c]

        return {self.my_attribute.name: retVal}

    def evaluate(self, **input):
        exec_ind = None
        for attr in self.attr_list:
            if attr.name not in input:
                raise Exception("Missing value for %s" % attr.name)

            ind = []
            for a_val in np.array([input[attr.name]]).flatten():
                inter_ind = (
                    np.argwhere(self.rules[attr.name] == a_val).flatten()
                    if a_val != "*"
                    else list(range(len(self.rules[attr.name])))
                )
                ind = np.union1d(ind, inter_ind).astype(int)

            if exec_ind is None:
                exec_ind = ind
            else:
                exec_ind = np.intersect1d(exec_ind, ind).flatten()

        if len(exec_ind) < 1:
            raise Exception(
                "Wrong number of rules executed %s for rule %s" % (exec_ind, self.name)
            )
        return {self.my_attribute.name: np.unique(self.output_values[exec_ind])}

    #     def add_rules(self,rule):
    #         pass

    @staticmethod
    def parse(node, attributes):
        name = node.findall("NAME")[0].text.strip()
        obj = DEXFunction(name, attributes[name])
        attr_list = node.findall("ATTRLIST")[0].text.split(";")
        a_list = []
        rules = {}

        for a in attr_list:
            a = a.strip()
            a_list.append(attributes[a])
            rules[a] = []

        for rule in node.findall("RULE"):
            cond = rule.findall("CONDITION")[0].text.split(";")
            result = rule.findall("RESULT")[0].text
            for i, c in enumerate(cond):
                rules[a_list[i].name].append(c)

            obj.output_values.append(result)

        for k, v in rules.items():
            rules[k] = np.array(v)

        obj.attr_list = a_list
        for a in obj.attr_list:
            a.set_function(obj)

        obj.rules = rules
        obj.output_values = np.array(obj.output_values)
        obj.rules_to_QQ()
        obj.kcnc()
        return obj


class DEXModel:
    def __init__(self, filename):

        tree = ET.parse(filename)  # ('Evaluation1.xml')
        root = tree.getroot()

        self.scales = {}
        self.attributes = {}
        self.functions = {}
        # parse scale
        for node in root.findall("SCALE"):
            obj = Scale.parse(node)
            self.scales[obj.name] = obj

        # parse attributes
        for node in root.findall("ATTRIBUTE"):
            obj = Attribute.parse(node, self.scales)
            self.attributes[obj.name] = obj

        # set parents of attributes
        for k, a in self.attributes.items():
            a.set_parent(self.attributes)

        # parse functions
        for node in root.findall("FUNCTION"):
            obj = DEXFunction.parse(node, self.attributes)
            self.functions[obj.name] = obj

        self.__post_process()

    def __post_process(self):
        for k, v in self.attributes.items():
            self.set_attribute_level(v)

        for k, v in self.functions.items():
            v.set_level()

    # set attribute levels
    #     input_attrs = []
    def set_attribute_level(self, node):
        if not node.child_attrs:
            node.level = 1
            return 1

        l = [self.set_attribute_level(a) for a in node.child_attrs]

        node.level = max(l) + 1
        return node.level

    def get_intput_attributes(self):
        retval = {}
        for k, v in self.attributes.items():
            if v.level == 1:
                retval[v.name] = list(v.scale.scalevalue.keys())

        return retval

    def evaluate_model(self, data, data_qq=None, skip_qq=False):
        def sort_by_level(obj):
            return obj.level

        x = sorted(self.functions.values(), key=sort_by_level)

        in_data = data
        for a in x:
            res = a.evaluate(**in_data)
            in_data = {**in_data, **res}
        if skip_qq:
            return in_data, None
            
        qq_data = data_qq
        if data_qq is None:
            qq_data = {}
            for k in data:
                if data[k] != "*":
                    qq_data[k] = self.attributes[k].map_qq(data[k])
                else:
                    qq_data[k] = data[k]

        for a in x:
            res = a.evaluate_QQ(**qq_data)
            qq_data = {**qq_data, **res}

        return in_data, qq_data
