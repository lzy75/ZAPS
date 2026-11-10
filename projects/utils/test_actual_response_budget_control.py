import unittest
from utils import actual_response_budget_control as rules


class Tests(unittest.TestCase):
    def test_costs_are_explicit_and_not_conflated(self):
        self.assertEqual(rules.expected_cost('ZAPS300'),dict(forwards=300,optimizer_backward=10,input_vjp=0,input_jvp=0))
        self.assertEqual(rules.expected_cost('ZAPS450')['forwards'],450)
        for arm in rules.RESPONSE_ARMS:
            row=rules.expected_cost(arm);self.assertEqual((row['forwards'],row['input_vjp'],row['input_jvp']),(445,29,29))


if __name__=='__main__':unittest.main()
