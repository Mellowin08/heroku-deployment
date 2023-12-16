class Footer extends HTMLElement{
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = `
        <footer class="py-4">
        <div class="container">
            <div class="row">
                <div class="col-md-4">
                    <h2 class="navbar-brand">RevU's</h2>
                </div>
                <div class="col-md-4">
                    <h5>Relevant Links:</h5>
                    <ul class="list-unstyled">
                        <li><a href="https://nijianmo.github.io/amazon/index.html#subsets">Data Source</a></li>
                        <li><a href="https://github.com/Mellowin08/CS-THESIS/">Website Repository</a></li>
                        <li><a href="../static/documents/research_paper.pdf">Research Paper</a></li>
                    </ul>
                </div>
                <div class="col-md-4">
                    <h5>Image Attribution:</h5>
                        <ul class="list-unstyled">
                                <li><a href="https://www.freepik.com/author/pikisuperstar">pikisuperstar</a></li>
                                <li><a href="https://storyset.com/">Storyset</a></li>
                                <li><a href="https://www.pexels.com/@olly/">Andrea Piacquadio</a></li>
                                <li><a href="https://www.pexels.com/@polina-tankilevitch/">Polina Tankilevitch</a></li>
                            </ul>
                        </li>
                    </ul>
                </div>
            </div>
            <div class="row">
                <div class="col-md-12 text-center">
                    <p>
                        This CS Thesis Writing is presented to the Faculty of the College of Computer Studies, AMA Computer College
                        <br>
                        In Partial Fulfillment of the Requirements of the Degree of Bachelor of Science in Computer Science (BSCS)
                    </p>
                </div>
            </div>
        </div>
    </footer>
        `
    }
}
customElements.define('footer-component', Footer);
