import numpy as np


class Polynomial:
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def from_string(string):
        # remove spaces between +/- signs
        string = string.replace("+ ", "").replace("- ", "-")
        monomials = string.split(" ")
        coeffs = []
        degrees = []

        for i in range(len(monomials)):
            # Extract degree from term
            term = monomials[i]
            if "x" not in term:
                degrees.append(0)
            else:
                if "^" not in term:
                    degrees.append(1)
                else:
                    degrees.append(int(term.split("^")[1]))

            # Get coefficient from term
            if "*" in term:
                coeffs.append(int(term.split("*")[0]))
            else:
                if "x" in term:
                    if term[0] == "x":
                        coeffs.append(1)
                    elif term[0] == "-":
                        coeffs.append(-1)
                    else:
                        coeffs.append(int(term.split("x")[0]))
                else:
                    coeffs.append(int(term))

        # sort coefficients by degree
        coeffs1 = np.array(coeffs)
        degrees1 = np.array(degrees)
        inds = degrees1.argsort()
        sortedcoeffs = coeffs1[inds]
        sorteddegrees = np.sort(degrees1)

        # need to go through terms and add coefficients where there are gaps in degrees between terms
        polyorder = max(sorteddegrees)
        newcoeffs = np.empty((1, polyorder+1))
        for i in range(polyorder+1):
            # check if there is a term of degree i in polynomial
            # if there isn't then add a 0 else add the coefficient
            if i in sorteddegrees:
                degree = sortedcoeffs[np.where(sorteddegrees == i)]
                newcoeffs[0][i] = degree
            else:
                newcoeffs[0][i] = 0

        return Polynomial(newcoeffs)

    def __repr__(self):
        polystring = ""
        print(len(self.coefficients[0]))
        for i in range(len(self.coefficients[0])):

            if self.coefficients[0][i] != 0:
                if i == 0:
                    polystring = polystring + str(int(self.coefficients[0][i]))
                if i == 1:
                    if self.coefficients[0][i] == 1:
                        polystring = polystring + "x"
                    else:
                        polystring = polystring + \
                            str(int(self.coefficients[0][i])) + "x"
                if i > 1:
                    if self.coefficients[0][i] == 1:
                        polystring = polystring + "x^" + str(i)
                    else:
                        polystring = polystring + \
                            str(int(self.coefficients[0][i]))+"*x^" + str(i)

                if i < len(self.coefficients[0])-1:
                    polystring = polystring + " " + "+" + " "

        return polystring

    def __eq__(self, other):

        size1 = np.size(self.coefficients)
        size2 = np.size(other.coefficients)

        if size1 > size2:
            other.coefficients = np.pad(
                other.coefficients, [(0, 0), (0, size1-size2)], "constant")
        if size2 > size1:
            self.coefficients = np.pad(
                self.coefficients, [(0, 0), (0, size2-size1)], "constant")

        if np.array_equal(self.coefficients, other.coefficients):
            return True

    def __add__(self, other):
        size1 = np.size(self.coefficients)
        size2 = np.size(other.coefficients)

        if size1 == size2:
            addedcoeffs = self.coefficients + other.coefficients
        else:
            if size1 > size2:
                other.coefficients = np.pad(
                    other.coefficients, [(0, 0), (0, size1-size2)], "constant")
            else:
                self.coefficients = np.pad(
                    self.coefficients, [(0, 0), (0, size2-size1)], "constant")

            addedcoeffs = np.add(self.coefficients, other.coefficients)

        return Polynomial(addedcoeffs)

    def __sub__(self, other):
        size1 = np.size(self.coefficients)
        size2 = np.size(other.coefficients)

        if size1 == size2:
            subbedcoeffs = self.coefficients - other.coefficients
        else:
            if size1 > size2:
                other.coefficients = np.pad(
                    other.coefficients, [(0, 0), (0, size1-size2)], "constant")
            else:
                self.coefficients = np.pad(
                    self.coefficients, [(0, 0), (0, size2-size1)], "constant")

            subbedcoeffs = self.coefficients - other.coefficients

        return Polynomial(subbedcoeffs)

    def __mul__(self, other):
        multcoeffs = np.polymul(self.coefficients[0], other.coefficients[0])
        return Polynomial(np.array([multcoeffs]))

    def __truediv__(self, other):

        return RationalPolynomial(self, other)


class RationalPolynomial:
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator

    @staticmethod
    def from_string(self, string):
        num = string.split("/")[0]
        denom = string.split("/")[1]
        num = num.replace("(", "").replace(")", "")
        denom = denom.replace("(", "").replace(")", "")

    def __repr__(self):
        polystring1 = ""
        for i in range(len(self.numerator.coefficients[0])):

            if self.numerator.coefficients[0][i] != 0:
                if i == 0:
                    polystring1 = polystring1 + \
                        str(int(self.numerator.coefficients[0][i]))
                if i == 1:
                    if self.numerator.coefficients[0][i] == 1:
                        polystring1 = polystring1 + "x"
                    else:
                        polystring1 = polystring1 + \
                            str(int(self.numerator.coefficients[0][i])) + "x"
                if i > 1:
                    if self.numerator.coefficients[0][i] == 1:
                        polystring1 = polystring1 + "x^" + str(i)
                    else:
                        polystring1 = polystring1 + \
                            str(int(
                                self.numerator.coefficients[0][i]))+"*x^" + str(i)

                if i < len(self.numerator.coefficients[0])-1:
                    polystring1 = polystring1 + " " + "+" + " "

        polystring2 = ""
        for i in range(len(self.denominator.coefficients[0])):

            if self.denominator.coefficients[0][i] != 0:
                if i == 0:
                    polystring2 = polystring2 + \
                        str(int(self.denominator.coefficients[0][i]))
                if i == 1:
                    if self.denominator.coefficients[0][i] == 1:
                        polystring2 = polystring2 + "x"
                    else:
                        polystring2 = polystring2 + \
                            str(int(self.denominator.coefficients[0][i])) + "x"
                if i > 1:
                    if self.denominator.coefficients[0][i] == 1:
                        polystring2 = polystring2 + "x^" + str(i)
                    else:
                        polystring2 = polystring2 + \
                            str(int(
                                self.denominator.coefficients[0][i]))+"*x^" + str(i)
                if i < len(self.denominator.coefficients[0])-1:
                    polystring2 = polystring2 + " " + "+" + " "
                # if i <= len(self.denominator.coefficients):
                #     polystring2 = polystring2 + " " + "+" + " "

        polystring = "(" + polystring1 + ")" + "/" + "(" + polystring2 + ")"
        return polystring

    def __eq__(self, other):
        if self.numerator == other.numerator:
            if self.denominator == other.denominator:
                return True
        else:
            return False

    def __add__(self, other):
        numerator = self.numerator * other.denominator + self.denominator*other.numerator
        denominator = self.denominator * other.denominator
        return RationalPolynomial(numerator, denominator)

    def __sub__(self, other):
        numerator = self.numerator * other.denominator - self.denominator*other.numerator
        denominator = self.denominator * other.denominator
        return RationalPolynomial(numerator, denominator)

    def __mul__(self, other):
        numerator = self.numerator * other.numerator
        denominator = self.denominator * other.denominator
        return RationalPolynomial(numerator, denominator)

    def __truediv__(self, other):
        numerator = self.numerator * other.denominator
        denominator = self.denominator * other.denominator
        return RationalPolynomial(numerator, denominator)
