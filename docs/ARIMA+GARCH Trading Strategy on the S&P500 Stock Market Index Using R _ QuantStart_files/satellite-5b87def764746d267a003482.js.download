_satellite.pushBlockingScript(function(event, target, $variables){
  pendoData = {
  visitor: {},
  account: {}
};

if(pageData) {
  // visitor
  if(pageDataTracker) {
    pendoData.visitor.id = pageDataTracker.getVisitorId();
  }
  if(pageData.visitor && pageData.visitor.userId) {
    pendoData.visitor.webuserID = pageData.visitor.userId;
  }
  if(pageData.visitor && pageData.visitor.accessType) {
    pendoData.visitor.accessType = pageData.visitor.accessType;
  }
  if(pageData.page && pageData.page.name && pageData.page.productName) {
    pendoData.visitor.pageName = pageData.page.productName + ':' + pageData.page.name;
  }
  if(pageData.page && pageData.page.type) {
    pendoData.visitor.pageType = pageData.page.type;
  }
  if(pageData.page && pageData.page.productName) {
    pendoData.visitor.pageProduct = pageData.page.productName;
  }
  if(pageData.page && pageData.page.language) {
    pendoData.visitor.pageLanguage = pageData.page.language;
  }
  if(pageData.page && pageData.page.environment) {
    pendoData.visitor.pageEnvironment = pageData.page.environment;
  }

  // account
  if(pageData.visitor && pageData.visitor.accountId) {
    pendoData.account.id = pageData.visitor.accountId;
  }
  if(pageData.visitor && pageData.visitor.accountName) {
    pendoData.account.name = pageData.visitor.accountName;
  }  
}

if(pendoData.visitor.id && pendoData.visitor.pageName && pendoData.visitor.pageProduct) {
  pendo.initialize(pendoData);
  _satellite.notify('pendo initialized with pageData.');
} else {
  _satellite.notify('couldn\'t initialize pendo.');
}

});
