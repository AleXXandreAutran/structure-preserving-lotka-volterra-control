from pd_network import run_checks


def test_weighted_cycle_accounting_and_resolvent():
    report = run_checks()
    assert len(report['rows']) == 4
    assert max((r['mass_identity_error'] for r in report['rows'])) < 1e-12
    large = report['rows'][-1]
    assert large['eight_step_flux_harvest'] < large['initial_mass']
    assert large['eight_step_old_state_reward'] > 10 * large['initial_mass']
