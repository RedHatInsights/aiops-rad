from setuptools import setup


setup(
    name="rad",
    version="0.9",
    description="AI-Ops Red Hat Anomaly Detection (RAD)",
    author="Parsa Hosseini, Ph.D.",
    author_email="phossein@redhat.com",
    url="https://github.com/ManageIQ/aiops-rad",
    packages=["rad"],
    install_requires=["numpy>=1.14",
                      "matplotlib",
                      "pandas",
                      "pyarrow",
                      "requests",
                      "s3fs",
                      "urllib3"],
    tests_require=['pytest',
                   'pytest-cov'],
    setup_requires=["pytest-runner"],
)
