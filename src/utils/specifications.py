from collections import namedtuple, OrderedDict

AssociationTestSpecification = namedtuple('AssociationTestSpecification', ['X', 'Y', 'A', 'B'])

ASSOCIATION_TESTS = OrderedDict(
    [
        (
            'Age',
            AssociationTestSpecification('age/young', 'age/old', 'valence/pleasant', 'valence/unpleasant')
        ),
        (
            'Arab-Muslim',
            AssociationTestSpecification('arab-muslim/other-people', 'arab-muslim/arab-muslim', 'valence/pleasant', 'valence/unpleasant')
        ),
        (
            'Asian',
            AssociationTestSpecification('asian/european-american', 'asian/asian-american', 'asian/american', 'asian/foreign')
        ),
        (
            'Disability',
            AssociationTestSpecification('disabled/disabled', 'disabled/abled', 'valence/pleasant', 'valence/unpleasant')
        ),
        (
            'Gender-Career',
            AssociationTestSpecification('gender/male', 'gender/female', 'gender/career', 'gender/family')
        ),
        (
            'Gender-Science',
            AssociationTestSpecification('gender/male', 'gender/female', 'gender/science', 'gender/liberal-arts')
        ),
        (
            'Insect-Flower',
            AssociationTestSpecification('insect-flower/flower', 'insect-flower/insect', 'valence/pleasant', 'valence/unpleasant')
        ),
        (
            'Native',
            AssociationTestSpecification('native/euro', 'native/native', 'native/us', 'native/world')
        ),
        (
            'Race',
            AssociationTestSpecification('race/european-american', 'race/african-american', 'valence/pleasant', 'valence/unpleasant')
        ),
        (
            'Religion',
            AssociationTestSpecification('religion/christianity', 'religion/judaism', 'valence/pleasant', 'valence/unpleasant')
        ),
        (
            'Sexuality',
            AssociationTestSpecification('sexuality/gay', 'sexuality/straight', 'valence/pleasant', 'valence/unpleasant')
        ),
        (
            'Skin-Tone',
            AssociationTestSpecification('skin-tone/light', 'skin-tone/dark', 'valence/pleasant', 'valence/unpleasant')
        ),
        (
            'Weapon',
            AssociationTestSpecification('weapon/white', 'weapon/black', 'weapon/tool', 'weapon/weapon')
        ),
        (
            'Weapon (Modern)',
            AssociationTestSpecification('weapon/white', 'weapon/black', 'weapon/tool-modern', 'weapon/weapon-modern')
        ),
        (
            'Weight',
            AssociationTestSpecification('weight/thin', 'weight/fat', 'valence/pleasant', 'valence/unpleasant')
        ),
    ]
)
