from unittest import TestCase

from utils.graph_utils import from_demangled_name_to_name


class Test(TestCase):
    def test_from_demangled_name_to_name(self):
        self.assertEqual(from_demangled_name_to_name('__gmon_start'), 'gmonstart')
        self.assertEqual(from_demangled_name_to_name('BINARYSEARCH_COUNTOCCURENCIES1_Test::TestBody()'), 'TestBody')
        self.assertEqual(from_demangled_name_to_name('testing::internal::TestFactoryImpl<BINARYSEARCH_SEARCHEND_Test>::~TestFactoryImpl()'), '~TestFactoryImpl')
        self.assertEqual(from_demangled_name_to_name('testing::internal::TestFactoryImpl<BINARYSEARCH_SEARCHEND_Test>::~TestFactoryImpl(char*)'), '~TestFactoryImpl')
        self.assertEqual(from_demangled_name_to_name('.__fxstat64'), 'fxstat64')



