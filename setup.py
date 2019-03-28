from setuptools import setup


setup(
    name="rad",
    version="0.8.2",
    description="AI-Ops Red Hat Anomaly Detection (RAD)",
    author="Parsa Hosseini, Ph.D.",
    author_email="phossein@redhat.com",
    url="https://gitlab.cee.redhat.com/phossein/rad",
    packages=["rad"],
    install_requires=["numpy",
                      "pandas",
                      "pyarrow",
                      "requests",
                      "s3fs",
                      "urllib3"],
)
