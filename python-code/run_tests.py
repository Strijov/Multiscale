import unittest

testmodules = ['UnitTests.test_iot_data', 'UnitTests.test_reg_matrix', 'UnitTests.test_ts_struct']
suite = unittest.TestSuite()


for t in testmodules:
    try:
        # If the module defines a suite() function, call it to get the suite.
        mod = __import__(t, globals(), locals(), ['suite'])
        suitefn = getattr(mod, 'suite')
        suite.addTest(suitefn())
    except (ImportError, AttributeError):
        # else, just load all the test cases from the module.
        suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))

    # suite.addTest(unittest.defaultTestLoader.loadTestsFromName('UnitTests.test_iot_data'))

unittest.TextTestRunner().run(suite)